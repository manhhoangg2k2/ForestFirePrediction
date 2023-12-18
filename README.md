//Code Arduino

#include <WiFi.h>
#include <HTTPClient.h>
#include <DHT.h>

const char* ssid = "Hoang Anh";
const char* password = "manhhoang.";
const char* serverUrl = "http://192.168.1.37:5000/data";

#define DHTPIN 13       // Chân D13 của ESP32
#define DHTTYPE DHT11   // Sử dụng cảm biến DHT11
DHT dht(DHTPIN, DHTTYPE);

const int airQualityPin = 2;   // Chân D2 của ESP32
const int soilMoisturePin = 3; // Chân D3 của ESP32

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }

  Serial.println("Connected to WiFi");
  dht.begin();
}

void loop() {
  // Đặt thông tin HTTP header
  HTTPClient http;
  http.begin(serverUrl);
  http.addHeader("Content-Type", "application/x-www-form-urlencoded");
  
  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();
  int airQualityValue = analogRead(airQualityPin);
  int soilMoistureValue = analogRead(soilMoisturePin);

  String postData = "temperature=" + String(temperature) + "&humidity=" + String(humidity) + "&air_quality=" + String(airQualityValue) + "&soil_moisture=" + String(soilMoistureValue);
  int httpResponseCode = http.POST(postData);

  // Kiểm tra mã phản hồi từ máy chủ HTTP
  if (httpResponseCode > 0) {
    Serial.print("HTTP Response code: ");
    Serial.println(httpResponseCode);
  } else {
    Serial.print("Error sending POST request. Error code: ");
    Serial.println(httpResponseCode);
  }
  delay(60000);
  // Giải phóng tài nguyên
  http.end();
}
