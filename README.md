# Anomaly Detection API

This API provides anomaly detection using an AutoEncoder model trained on normal data.
It is designed for integration into monitoring, quality control, and anomaly detection pipelines.

---

## Overview

- **Base URL:** `http://<host>:8000`
- **API Title:** Anomaly Detection API
- **Version:** 1.0.0
- **Protocol:** HTTP/REST + JSON
- **Authentication:** None (can be extended with API keys or OAuth2)

---

## Endpoint Summary

### POST `/anomaly` — Detect Anomaly

- **Description:**  
  Computes an anomaly score for the given numeric input and returns both the score and a status label.

- **Tag:** `Detect Anomaly`  
- **OperationId:** `detect_anomaly`  
- **Method:** `POST`  
- **Content-Type:** `application/json`  
- **Consumes:** `application/json`  
- **Produces:** `application/json`  

---

## Request Specification

### URL

```text
POST /anomaly# Anomaly Detection API

This API provides anomaly detection using an AutoEncoder model trained on normal data.
It is designed for integration into monitoring, quality control, and anomaly detection pipelines.

---

## Overview

- **Base URL:** `http://<host>:8000`
- **API Title:** Anomaly Detection API
- **Version:** 1.0.0
- **Protocol:** HTTP/REST + JSON
- **Authentication:** None (can be extended with API keys or OAuth2)

---

## Endpoint summary

### POST `/anomaly` — Detect Anomaly

- **Description:**  
  Computes an anomaly score for the given numeric input and returns both the score and a status label.

- **Tag:** `Detect Anomaly`  
- **OperationId:** `detect_anomaly`  
- **Method:** `POST`  
- **Content-Type:** `application/json`  
- **Consumes:** `application/json`  
- **Produces:** `application/json`  

---

## Request specification

### URL

```text
POST /anomaly
