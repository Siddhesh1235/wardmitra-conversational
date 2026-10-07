-- Migration: Add idempotency_key to complaints table
-- TODO 2026-10-06T15:50:12+05:30

ALTER TABLE complaints ADD COLUMN IF NOT EXISTS idempotency_key VARCHAR(255);

CREATE UNIQUE INDEX IF NOT EXISTS idx_complaints_user_idempotency_key 
ON complaints(user_id, idempotency_key) 
WHERE idempotency_key IS NOT NULL;
