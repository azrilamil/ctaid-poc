// Convex functions for the CTAID POC demo.
// Called from the Python app through the Convex HTTP API (see app/backend.py).
// Every write is synthetic demo data. No user accounts, no clinical data.

import { mutation, query } from "./_generated/server";
import { v } from "convex/values";

const now = () => new Date().toISOString();

export const saveDatasetMeta = mutation({
  args: {
    meta: v.object({
      name: v.string(),
      rows: v.number(),
      columns: v.number(),
      records: v.number(),
      class_counts: v.optional(v.record(v.string(), v.number())),
      digest: v.optional(v.string()),
    }),
  },
  handler: async (ctx, { meta }) => {
    const doc = {
      name: meta.name,
      rows: meta.rows,
      columns: meta.columns,
      records: meta.records,
      classCounts: meta.class_counts ?? {},
      digest: meta.digest ?? "",
      createdAt: now(),
    };
    const id = await ctx.db.insert("datasetMeta", doc);
    return { id, ...doc };
  },
});

export const latestDatasetMeta = query({
  args: {},
  handler: async (ctx) => {
    return await ctx.db.query("datasetMeta").withIndex("by_created").order("desc").first();
  },
});

export const saveModelVersion = mutation({
  args: {
    record: v.object({
      model: v.string(),
      label: v.optional(v.string()),
      metrics: v.record(v.string(), v.number()),
      rows_used: v.number(),
      feature_count: v.number(),
      artifact: v.optional(v.string()),
    }),
  },
  handler: async (ctx, { record }) => {
    const doc = {
      model: record.model,
      label: record.label ?? record.model,
      metrics: record.metrics,
      rowsUsed: record.rows_used,
      featureCount: record.feature_count,
      artifact: record.artifact,
      createdAt: now(),
    };
    const id = await ctx.db.insert("modelVersions", doc);
    return { id, ...doc };
  },
});

export const listModelVersions = query({
  args: { limit: v.optional(v.number()) },
  handler: async (ctx, { limit }) => {
    const rows = await ctx.db
      .query("modelVersions")
      .withIndex("by_created")
      .order("desc")
      .take(Math.min(limit ?? 20, 100));
    return rows.map((row) => ({
      model: row.model,
      label: row.label,
      metrics: row.metrics,
      rows_used: row.rowsUsed,
      feature_count: row.featureCount,
      artifact: row.artifact,
      created_at: row.createdAt,
    }));
  },
});

export const savePrediction = mutation({
  args: {
    record: v.object({
      source: v.string(),
      predicted_class: v.number(),
      predicted_label: v.string(),
      confidence: v.number(),
      probabilities: v.record(v.string(), v.number()),
      top_features: v.array(v.string()),
    }),
  },
  handler: async (ctx, { record }) => {
    const doc = {
      source: record.source,
      predictedClass: record.predicted_class,
      predictedLabel: record.predicted_label,
      confidence: record.confidence,
      probabilities: record.probabilities,
      topFeatures: record.top_features,
      createdAt: now(),
    };
    const id = await ctx.db.insert("predictions", doc);
    return { id, ...doc };
  },
});

export const listPredictions = query({
  args: { limit: v.optional(v.number()) },
  handler: async (ctx, { limit }) => {
    const rows = await ctx.db
      .query("predictions")
      .withIndex("by_created")
      .order("desc")
      .take(Math.min(limit ?? 50, 200));
    return rows.map((row) => ({
      source: row.source,
      predicted_class: row.predictedClass,
      predicted_label: row.predictedLabel,
      confidence: row.confidence,
      probabilities: row.probabilities,
      top_features: row.topFeatures,
      created_at: row.createdAt,
    }));
  },
});

export const countPredictions = query({
  args: {},
  handler: async (ctx) => {
    const rows = await ctx.db.query("predictions").take(500);
    return rows.length;
  },
});

// Demo reset: clears prediction history and model versions.
// The daily reseed script also calls this.
export const resetDemo = mutation({
  args: {},
  handler: async (ctx) => {
    let removed = 0;
    for (const table of ["predictions", "modelVersions"]) {
      const rows = await ctx.db.query(table).take(1000);
      for (const row of rows) {
        await ctx.db.delete(row._id);
        removed += 1;
      }
    }
    return { removed };
  },
});
