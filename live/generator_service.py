import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng import ControlVector, ControlModel
from dataclasses import dataclass
from typing import Optional, List, Tuple

@dataclass
class GeneratorConfig:
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    batch_size: int = 32
    method: str = "pca_center"
    repetition_penalty: float = 1.5
    temperature: float = 0.7
    n_context: int = 50
    n_repeat_window: int = 100
    min_repeat_penalty: float = 1.5
    max_repeat_penalty: float = 2.5
    layers: list[int] = None
    device: str = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    allow_eof: bool = True  # Control whether to allow EOS token
    
    def __post_init__(self):
        if self.layers is None:
            self.layers = list(range(5, 22))

@dataclass
class Token:
    content: str
    x_strength: float
    y_strength: float
    
    def __post_init__(self):
        # Ensure values are within -1 to 1 range
        self.x_strength = max(-1, min(1, self.x_strength))
        self.y_strength = max(-1, min(1, self.y_strength))

# LLAMA 3.1 Instruct
START_HEADER = "<|start_header_id|>"
END_HEADER = "<|end_header_id|>"
EOT = "<|eot_id|>"
BEGIN_TEXT = "<|begin_of_text|>"

def parse_chat_message(message: str, role: str = "user") -> str:
    if role not in ["user", "assistant", "system"]:
        raise ValueError("Role must be 'user', 'assistant', or 'system'")
    return f"{START_HEADER}{role}{END_HEADER}\n\n{message}{EOT}"

def format_chat_sequence(messages: List[Tuple[str, str]], add_assistant_prefix: bool = True) -> str:
    template = []
    for role, content in messages:
        template.append(parse_chat_message(content, role))
    if add_assistant_prefix and messages and messages[-1][0] != "assistant":
        template.append(f"{START_HEADER}assistant{END_HEADER}\n\n")
    return "".join(template)

def unparse_chat_message(formatted_text: str) -> List[Tuple[str, str]]:
    text = formatted_text.strip()
    if text.startswith(BEGIN_TEXT): text = text[len(BEGIN_TEXT):]
    messages = []
    parts = text.split(START_HEADER)
    if parts[0].strip() == "": parts = parts[1:]
    for part in parts:
        try:
            role_content = part.split(END_HEADER, 1)
            if len(role_content) == 1: role, content = role_content[0].strip(), ""
            else: role, content = role_content
            content = content.split(EOT)[0].strip()
            role = role.strip()
            if role: messages.append((role, content))
        except Exception:
            continue
    return messages

class Generator:
    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or GeneratorConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.tokenizer.pad_token_id = 0
        
        model = (AutoModelForCausalLM
                .from_pretrained(self.config.model_name, torch_dtype=torch.float16)
                .to(self.config.device))
        self.model = ControlModel(model, self.config.layers)
        
        # Initialize with default vectors
        self.load_vectors("TPPAST", "TPPRESENT")
        
        # Initialize conversation state
        self.message_history: List[Tuple[str, str]] = []
        self.current_response = ""
        # Store original start messages for reset
        self.original_system_message = "Pretend to be Tom Petty. You are singing a song."
        self.original_user_message = "I'm Running down a dream.. "
        self.add_message(self.original_system_message, "system")
        # Add initial user message to trigger token initialization
        self.add_message(self.original_user_message, "user")
        
        # Current vector strengths
        self.x_strength = 0.1
        self.y_strength = 0.1
        
    def load_vectors(self, x_vector: str, y_vector: str) -> None:
        """Load control vectors from files"""
        try:
            # Load vectors from files
            self.x_vector = ControlVector.import_gguf(f"vec/{x_vector}.gguf")
            self.y_vector = ControlVector.import_gguf(f"vec/{y_vector}.gguf")
            
            # Store current vector names
            self.x_vector_name = x_vector
            self.y_vector_name = y_vector
            
            # Reset control
            self.update_controls(0, 0)
            
            print(f"Loaded vectors: {x_vector} (X) and {y_vector} (Y)")
        except Exception as e:
            raise Exception(f"Error loading vectors: {str(e)}")
    
    def add_message(self, message: str, role: str = "user") -> None:
        """Add a new message to the conversation history"""
        self.message_history.append((role, message))
        if role == "user":
            # Format entire conversation and set as new context
            formatted = format_chat_sequence(self.message_history)
            self.reset_tokens(formatted)
            self.current_response = ""
        elif role == "assistant":
            self.current_response = message
            
    def reset_conversation(self, initial_prompt: str = "Who are you?") -> None:
        """Reset the entire conversation history and start fresh"""
        self.message_history = []
        self.current_response = ""
        self.add_message(initial_prompt, "user")
    
    def reset_to_original(self) -> None:
        """Reset to the original start messages"""
        self.message_history = []
        self.current_response = ""
        self.add_message(self.original_system_message, "system")
        self.add_message(self.original_user_message, "user")
        print("  🔄 Reset generator to original start messages")
        
    def reset_tokens(self, prompt: str) -> None:
        """Reset the token state with new prompt"""
        self.tokens = self.tokenizer.tokenize(prompt)
        self.token_history = []
        self.step = 0
        print(f"  Token state initialized with {len(self.tokens)} tokens")
        
    def update_controls(self, x_strength: float, y_strength: float) -> None:
        """Update control vector strengths"""
        self.x_strength = max(-1, min(1, x_strength))
        self.y_strength = max(-1, min(1, y_strength))
        
    def set_allow_eof(self, allow: bool) -> None:
        """Toggle whether to allow EOF tokens"""
        self.config.allow_eof = allow
        
    def calculate_repetition_penalty(self, token_id: int) -> float:
        """Calculate dynamic repetition penalty based on token frequency"""
        if not self.token_history:
            return 1.0
            
        # Count occurrences in recent history
        recent_tokens = self.token_history[-self.config.n_repeat_window:]
        count = recent_tokens.count(token_id)
        
        if count == 0:
            return 1.0
            
        # Scale penalty based on frequency
        base_penalty = self.config.min_repeat_penalty
        additional_penalty = (count - 1) * 0.5  # Increase penalty for each repeat
        penalty = min(base_penalty + additional_penalty, self.config.max_repeat_penalty)
        
        return penalty
        
    def next(self) -> Token:
        self.x_strength = self.x_strength * 0.75 # half strength
        self.y_strength = self.y_strength * 0.75 # half strength

        # Ensure tokens are initialized
        if not hasattr(self, 'tokens') or len(self.tokens) == 0:
            raise RuntimeError("Tokens not initialized. Call reset_tokens() or add_message() first.")
        
        current_response_tokens = len(self.current_response.split())
        if current_response_tokens >= 300:
            next_token_str = EOT
            self.add_message(self.current_response.strip(), "assistant")
            return Token(
                content=next_token_str,
                x_strength=self.x_strength,
                y_strength=self.y_strength,
            )

        x_vec = self.x_vector * self.x_strength
        y_vec = self.y_vector * self.y_strength
        combined_vec = x_vec + y_vec
        self.model.set_control(combined_vec)

        context = self.tokenizer.convert_tokens_to_string(self.tokens[-self.config.n_context:])
        model_tokens = self.tokenizer(context, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            logits = self.model.forward(**model_tokens).logits[0, -1, :]
        
        logits = logits / self.config.temperature
        
        if len(self.tokens) > 0:
            context_tokens = self.tokenizer.convert_tokens_to_ids(self.tokens[-self.config.n_repeat_window:])
            for token_id in range(len(logits)):
                penalty = self.calculate_repetition_penalty(token_id)
                if penalty > 1.0:
                    logits[token_id] = logits[token_id] / penalty
        
        if not self.config.allow_eof:
            logits[self.tokenizer.eos_token_id] = -float('inf')
        
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        
        self.token_history.append(next_token.item())
        if len(self.token_history) > self.config.n_repeat_window:
            self.token_history.pop(0)
        
        next_token_str = self.tokenizer.decode(next_token)
        self.tokens.append(next_token_str)
        self.current_response += next_token_str
        self.step += 1
        
        # Check for EOT token that should trigger a reset
        if EOT in next_token_str:
            # Reset to original start messages
            self.reset_to_original()

        return Token(
            content=next_token_str,
            x_strength=self.x_strength,
            y_strength=self.y_strength,
        ) 