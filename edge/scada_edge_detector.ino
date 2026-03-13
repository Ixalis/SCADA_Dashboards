/*
 * SCADA Anomaly Detection — Yolo UNO Edge Node
 * ================================================
 * Runs Isolation Forest inference on ESP32-S3 and sends
 * results to the gateway via HTTP/Wi-Fi.
 *
 * Hardware: OhStem Yolo UNO (ESP32-S3, 8MB PSRAM, 16MB Flash)
 * Model:    300-tree Isolation Forest trained on simulator data
 *
 * Setup:
 *   1. Copy scada_if_model.h to the same folder as this .ino
 *   2. Update WiFi credentials and gateway IP below
 *   3. Upload via Arduino IDE (Board: ESP32S3 Dev Module)
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include "scada_if_model.h"

// ============================================================================
// Configuration
// ============================================================================
const char* WIFI_SSID     = "Limitless Eternity";
const char* WIFI_PASSWORD = "TheEndOfAllThings";
const char* GATEWAY_URL   = "http://172.20.10.6:8000/detect";  // FastAPI server

const unsigned long SAMPLE_INTERVAL_MS = 1000;  // 1 Hz sampling
const int LED_PIN = 2;  // Built-in LED for alert indication

// ============================================================================
// Simulated sensor data (replace with real sensor reads in production)
// ============================================================================
float sensor_data[SCADA_IF_N_FEATURES];

// Simple simulator state for demo
float tank_levels[3]  = {500.0, 600.0, 400.0};
float flow_rates[6]   = {2.0, 1.8, 2.2, 1.9, 2.1, 2.0};

void generate_sample() {
    // Mean-reverting tank levels
    float targets[3] = {500.0, 600.0, 400.0};
    for (int i = 0; i < 3; i++) {
        tank_levels[i] += 0.05 * (targets[i] - tank_levels[i]);
        tank_levels[i] += random(-200, 200) / 100.0;  // noise
        tank_levels[i] = constrain(tank_levels[i], 50.0, 1000.0);
        sensor_data[i] = tank_levels[i];
    }

    // Flow rates
    for (int i = 0; i < 6; i++) {
        flow_rates[i] = 2.0 + random(-10, 10) / 100.0;
        flow_rates[i] = constrain(flow_rates[i], 0.0, 5.0);
        sensor_data[3 + i] = flow_rates[i];
    }

    // Analyzers (chemical measurements)
    float analyzer_means[9] = {250, 300, 350, 200, 220, 400, 7.0, 450, 480};
    for (int i = 0; i < 9; i++) {
        sensor_data[9 + i] = analyzer_means[i] + random(-100, 100) / 10.0;
    }

    // Pressure (correlated with flow)
    float avg_flow = 0;
    for (int i = 0; i < 6; i++) avg_flow += flow_rates[i];
    avg_flow /= 6.0;
    for (int i = 0; i < 3; i++) {
        sensor_data[18 + i] = 150 + 20 * avg_flow + random(-50, 50) / 10.0;
    }

    // Pump states (binary)
    for (int i = 0; i < 12; i++) {
        sensor_data[21 + i] = (i % 2 == 0) ? 1.0 : 0.0;
    }

    // Valve states
    for (int i = 0; i < 6; i++) {
        sensor_data[33 + i] = (i % 3 != 2) ? 1.0 : 0.0;
    }

    // Remaining sensors
    for (int i = 39; i < SCADA_IF_N_FEATURES; i++) {
        sensor_data[i] = 100.0 + random(-100, 100) / 10.0;
    }
}

// ============================================================================
// WiFi + HTTP
// ============================================================================
void setup_wifi() {
    Serial.print("Connecting to WiFi");
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 30) {
        delay(500);
        Serial.print(".");
        attempts++;
    }
    
    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("\nConnected! IP: %s\n", WiFi.localIP().toString().c_str());
    } else {
        Serial.println("\nWiFi connection failed — running in offline mode");
    }
}

void send_to_gateway(float score, bool is_anomaly) {
    if (WiFi.status() != WL_CONNECTED) return;

    HTTPClient http;
    http.begin(GATEWAY_URL);
    http.addHeader("Content-Type", "application/json");

    // Build JSON with first 10 sensor values + IF score
    String json = "{\"values\":[";
    for (int i = 0; i < 10; i++) {
        json += String(sensor_data[i], 4);
        if (i < 9) json += ",";
    }
    json += "],\"if_score\":" + String(score, 6);
    json += ",\"if_anomaly\":" + String(is_anomaly ? "true" : "false");
    json += ",\"source\":\"yolo_uno_edge\"}";

    int httpCode = http.POST(json);
    if (httpCode > 0) {
        // Serial.printf("  -> Gateway responded: %d\n", httpCode);
    } else {
        Serial.printf("  -> Gateway error: %s\n", http.errorToString(httpCode).c_str());
    }
    http.end();
}

// ============================================================================
// Main
// ============================================================================
void setup() {
    Serial.begin(115200);
    delay(1000);
    
    Serial.println();
    Serial.println("=================================");
    Serial.println("SCADA Edge Detector — Yolo UNO");
    Serial.printf("IF Model: %d trees, %d features\n", SCADA_IF_N_TREES, SCADA_IF_N_FEATURES);
    Serial.printf("Threshold: %.6f\n", SCADA_IF_THRESHOLD);
    Serial.println("=================================");
    
    pinMode(LED_PIN, OUTPUT);
    setup_wifi();
    
    randomSeed(analogRead(0));
}

void loop() {
    unsigned long start = millis();
    
    // Generate or read sensor data
    generate_sample();
    
    // Run Isolation Forest inference
    float score = scada_if_score(sensor_data);
    bool is_anomaly = score < SCADA_IF_THRESHOLD;
    
    unsigned long inference_ms = millis() - start;
    
    // LED indication
    digitalWrite(LED_PIN, is_anomaly ? HIGH : LOW);
    
    // Serial output
    Serial.printf("[%8lu] Score: %+.4f | %s | %lu ms | Tank: %.1f\n",
        millis(),
        score,
        is_anomaly ? "!! ANOMALY" : "   Normal ",
        inference_ms,
        sensor_data[0]
    );
    
    // Send to gateway
    send_to_gateway(score, is_anomaly);
    
    // Maintain sampling rate
    unsigned long elapsed = millis() - start;
    if (elapsed < SAMPLE_INTERVAL_MS) {
        delay(SAMPLE_INTERVAL_MS - elapsed);
    }
}
