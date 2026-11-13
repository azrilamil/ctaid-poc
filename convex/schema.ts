// Convex schema for the CTAID POC demo.
// Three tables only. Everything stored here is synthetic demo traffic:
// dataset metadata, model versions and prediction history. No accounts.

import { defineSchema, defineTable } from "convex/server";
import { v } from "convex/values";

export default defineSchema({
  datasetMeta: defineTable({
    name: v.string(),
    rows: v.number(),
    columns: v.number(),
    records: v.number(),
    classCounts: v.record(v.string(), v.number()),
    digest: v.string(),
    createdAt: v.string(),
  }).index("by_created", ["createdAt"]),

  modelVersions: defineTable({
    model: v.string(),
    label: v.string(),
    metrics: v.record(v.string(), v.number()),
    rowsUsed: v.number(),
    featureCount: v.number(),
    artifact: v.optional(v.string()),
    createdAt: v.string(),
  }).index("by_created", ["createdAt"]),

  predictions: defineTable({
    source: v.string(),
    predictedClass: v.number(),
    predictedLabel: v.string(),
    confidence: v.number(),
    probabilities: v.record(v.string(), v.number()),
    topFeatures: v.array(v.string()),
    createdAt: v.string(),
  }).index("by_created", ["createdAt"]),
});
