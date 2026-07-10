-- KPT2 Multi-Signal Kitchen Intelligence System
-- MySQL schema for storing simulated order lifecycles, merchant bias
-- profiles, and per-model predictions.
--
-- Design notes:
--   - `merchants` holds the slowly-changing Merchant Bias Index (MBI),
--     recomputed periodically by the batch pipeline.
--   - `orders` holds one row per simulated/observed order lifecycle.
--   - `predictions` is a narrow table (one row per model per order) so
--     new models can be added without an ALTER TABLE migration.
--   - `drift_flags` is an append-only audit log of merchants flagged by
--     the drift monitor, indexed for fast "give me all currently open
--     flags" queries.
--
-- All foreign keys are indexed implicitly by MySQL/InnoDB; additional
-- composite indexes below are added to match the pipeline's actual
-- query patterns (per-merchant time-range scans, per-model MAE rollups).

CREATE DATABASE IF NOT EXISTS kpt2 CHARACTER SET utf8mb4;
USE kpt2;

-- ---------------------------------------------------------------------
-- merchants
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS merchants (
    merchant_id     INT PRIMARY KEY,
    display_name    VARCHAR(120)      NOT NULL DEFAULT '',
    mbi_offset_min  DECIMAL(6, 3)     NOT NULL DEFAULT 0.000,
    mbi_updated_at  DATETIME          NOT NULL DEFAULT CURRENT_TIMESTAMP,
    is_flagged      BOOLEAN           NOT NULL DEFAULT FALSE,
    created_at      DATETIME          NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- ---------------------------------------------------------------------
-- orders
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS orders (
    order_id            BIGINT AUTO_INCREMENT PRIMARY KEY,
    merchant_id         INT           NOT NULL,
    order_time          DATETIME      NOT NULL,
    actual_ready_time   DATETIME      NOT NULL,
    for_time            DATETIME      NOT NULL,
    rider_arrival_time  DATETIME      NOT NULL,
    actual_kpt_min      DECIMAL(6, 2) NOT NULL,
    active_orders       SMALLINT      NOT NULL,
    complexity          DECIMAL(4, 2) NOT NULL,
    wait_cluster        DECIMAL(5, 4) NOT NULL,
    kls                 DECIMAL(5, 4) NOT NULL,
    for_adj_kpt_min     DECIMAL(6, 2) NOT NULL,
    historical_pattern  DECIMAL(6, 2) NOT NULL,
    data_split          ENUM('train', 'test') NOT NULL DEFAULT 'train',

    CONSTRAINT fk_orders_merchant
        FOREIGN KEY (merchant_id) REFERENCES merchants(merchant_id)
        ON DELETE CASCADE,

    INDEX idx_orders_merchant_time (merchant_id, order_time),
    INDEX idx_orders_time (order_time)
) ENGINE=InnoDB;

-- ---------------------------------------------------------------------
-- predictions  (narrow / long format: one row per model per order)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id   BIGINT AUTO_INCREMENT PRIMARY KEY,
    order_id        BIGINT        NOT NULL,
    model_name      ENUM('baseline', 'crs', 'shadow') NOT NULL,
    predicted_kpt   DECIMAL(6, 2) NOT NULL,
    abs_error       DECIMAL(6, 2) NOT NULL,
    created_at      DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_predictions_order
        FOREIGN KEY (order_id) REFERENCES orders(order_id)
        ON DELETE CASCADE,

    UNIQUE KEY uq_prediction_order_model (order_id, model_name),
    INDEX idx_predictions_model (model_name)
) ENGINE=InnoDB;

-- ---------------------------------------------------------------------
-- drift_flags  (append-only audit trail)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS drift_flags (
    flag_id         BIGINT AUTO_INCREMENT PRIMARY KEY,
    merchant_id     INT           NOT NULL,
    flag_type       ENUM('marking_instability', 'shadow_divergence') NOT NULL,
    metric_value    DECIMAL(8, 4) NOT NULL,
    flagged_at      DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_drift_flags_merchant
        FOREIGN KEY (merchant_id) REFERENCES merchants(merchant_id)
        ON DELETE CASCADE,

    INDEX idx_drift_flags_merchant (merchant_id, flagged_at),
    INDEX idx_drift_flags_type (flag_type)
) ENGINE=InnoDB;

-- ---------------------------------------------------------------------
-- Example rollup query the pipeline actually runs:
-- per-model MAE, using the indexed (model_name) column.
-- ---------------------------------------------------------------------
-- SELECT model_name, AVG(abs_error) AS mae, COUNT(*) AS n
-- FROM predictions
-- GROUP BY model_name;
