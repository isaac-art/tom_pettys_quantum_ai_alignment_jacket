#include <ArduinoBLE.h>
#include <LSM6DS3.h>

// BLE Service and Characteristics
BLEService imuService("12345678-1234-1234-1234-123456789abc");
BLECharacteristic imuDataChar("12345678-1234-1234-1234-123456789abd", BLERead | BLENotify, 24);

// IMU data structure
struct IMUData {
  float yaw;
  float pitch;
  float roll;
  float accelX;
  float accelY;
  float accelZ;
};

// Initialize LSM6DS3 sensor
LSM6DS3 myIMU(I2C_MODE, 0x6A);

IMUData imuData;
unsigned long lastUpdate = 0;
const unsigned long updateInterval = 50; // 20Hz

void setup() {
  Serial.begin(9600);
  
  // Initialize IMU
  if (myIMU.begin() != 0) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }
  Serial.println("IMU initialized successfully!");
  
  // Initialize BLE
  if (!BLE.begin()) {
    Serial.println("Starting BLE failed!");
    while (1);
  }
  
  // Set BLE device name and service
  BLE.setLocalName("XIAO_IMU_Controller");
  BLE.setDeviceName("XIAO_IMU_Controller");
  BLE.setAdvertisedService(imuService);
  
  // Add characteristics
  imuService.addCharacteristic(imuDataChar);
  BLE.addService(imuService);
  
  // Set initial values
  imuDataChar.writeValue((byte*)&imuData, sizeof(imuData));
  
  // Start advertising
  BLE.advertise();
  Serial.println("BLE device is now advertising as 'XIAO_IMU_Controller'");
}

void loop() {
  BLE.poll();
  
  // Update IMU data at specified interval
  if (millis() - lastUpdate >= updateInterval) {
    updateIMUData();
    sendIMUData();
    lastUpdate = millis();
  }
}

void updateIMUData() {
  float accelX, accelY, accelZ;
  float gyroX, gyroY, gyroZ;
  
  // Read acceleration data
  accelX = myIMU.readFloatAccelX();
  accelY = myIMU.readFloatAccelY();
  accelZ = myIMU.readFloatAccelZ();
  
  // Read gyroscope data
  gyroX = myIMU.readFloatGyroX();
  gyroY = myIMU.readFloatGyroY();
  gyroZ = myIMU.readFloatGyroZ();
  
  // Calculate pitch from accelerometer (more accurate for tilt)
  // Pitch = atan2(-accelX, sqrt(accelY*accelY + accelZ*accelZ)) * 180/PI
  float pitch = atan2(-accelX, sqrt(accelY*accelY + accelZ*accelZ)) * 180.0 / PI;
  
  // Simple gyro integration for yaw and roll
  static float yawAngle = 0.0;
  static float rollAngle = 0.0;
  
  // Integrate gyro data (assuming 50Hz update rate)
  float dt = 0.02; // 50Hz = 20ms
  yawAngle += gyroZ * dt;
  rollAngle += gyroX * dt;
  
  // Keep angles in -180 to +180 range
  while (yawAngle > 180) yawAngle -= 360;
  while (yawAngle < -180) yawAngle += 360;
  while (rollAngle > 180) rollAngle -= 360;
  while (rollAngle < -180) rollAngle += 360;
  
  imuData.yaw = yawAngle;
  imuData.pitch = pitch;
  imuData.roll = rollAngle;
  
  imuData.accelX = accelX;
  imuData.accelY = accelY;
  imuData.accelZ = accelZ;
}

void sendIMUData() {
  imuDataChar.writeValue((byte*)&imuData, sizeof(imuData));
  
  // Debug output
  Serial.print("Yaw: "); Serial.print(imuData.yaw, 1);
  Serial.print("°, Pitch: "); Serial.print(imuData.pitch, 1);
  Serial.print("°, Roll: "); Serial.print(imuData.roll, 1);
  Serial.print("°, Accel: "); Serial.print(imuData.accelX, 2);
  Serial.print(", "); Serial.print(imuData.accelY, 2);
  Serial.print(", "); Serial.println(imuData.accelZ, 2);
}

