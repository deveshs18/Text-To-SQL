# Prompt Improvements for Complex Queries

## Summary of Changes

Enhanced all prompt builders to handle complex queries that require:
- CTEs (WITH clauses)
- Window functions (ROW_NUMBER() OVER ...)
- Finding "most common" items per group
- Multiple aggregations with percentages
- Joining multiple CTEs

## Key Enhancements

### 1. **Complexity Detection** ✅
Added `is_complex_query()` function that detects:
- Keywords: "most common", "most frequent", "percentage", "for each", "per group"
- Multiple aggregations mentioned (count, average, percentage, etc.)
- Pattern: "most/top X per group" or "for each X, show Y and Z"

### 2. **Complex Examples** ✅
Added 3 comprehensive examples showing:
- CTEs with ROW_NUMBER() for finding top items
- Percentage calculations using AVG(CASE WHEN ...)
- Multiple aggregations with joins
- ROUND() for formatting

### 3. **Enhanced Prompt Builders**

#### **Few-Shot Prompt:**
- **Simple queries**: Uses 4 basic examples
- **Complex queries**: Uses 2-3 complex CTE examples + 1 simple reference
- Adds explicit guidance: "Use CTEs for multi-step logic"
- Shows ROW_NUMBER() pattern for "most common" items

#### **Chain-of-Thought (CoT):**
- **Simple queries**: 5-step reasoning
- **Complex queries**: Enhanced 5-step plan:
  1. Identify what needs to be calculated
  2. Plan CTE structure (step-by-step)
  3. Write CTEs with patterns
  4. Final SELECT with joins
  5. Generate complete SQL

#### **Least-to-Most (LtM):**
- **Simple queries**: 4 substeps (A-D)
- **Complex queries**: Enhanced substeps:
  - A: Identify all requirements
  - B: Plan CTE structure
  - C: Key SQL patterns (percentages, ROW_NUMBER(), formatting)
  - D: Write complete SQL with CTEs

#### **Execution-Guided (EG):**
- **Simple queries**: Direct SQL generation
- **Complex queries**: Includes explicit guidelines:
  - When to use CTEs
  - How to use ROW_NUMBER() OVER ...
  - How to calculate percentages
  - How to join CTEs

### 4. **Refinement Prompt** ✅
Enhanced to provide correction guidelines for complex queries:
- Suggests CTEs when appropriate
- Shows patterns for fixing common mistakes
- Guides toward proper CTE structure

## Example Query Matching

### Your Query:
```
"For each education level and gender, show how many people there are, 
the percentage earning more than 50K, the average age and hours worked 
per week, and the most common occupation, ranked by highest percentage 
of high earners"
```

### Detection:
- ✅ Detected as **COMPLEX** (has "for each", "percentage", "most common", multiple aggregations)
- ✅ Prompts will use complex examples
- ✅ All 4 techniques will guide toward CTE structure

### Expected Pattern in Generated SQL:
```sql
WITH occ_counts AS (
  SELECT education, sex, occupation, COUNT(*) AS occ_cnt
  FROM adult_income
  GROUP BY education, sex, occupation
),
top_occ AS (
  SELECT education, sex, occupation
  FROM (
    SELECT education, sex, occupation, occ_cnt,
           ROW_NUMBER() OVER (
             PARTITION BY education, sex
             ORDER BY occ_cnt DESC, occupation ASC
           ) AS rn
    FROM occ_counts
  )
  WHERE rn = 1
),
agg AS (
  SELECT
    education, sex,
    COUNT(*) AS total_people,
    AVG(CASE WHEN income = '>50K' THEN 1.0 ELSE 0.0 END) AS pct_high_income,
    AVG(age) AS avg_age,
    AVG(hours_per_week) AS avg_hours
  FROM adult_income
  GROUP BY education, sex
)
SELECT
  a.education, a.sex, a.total_people,
  ROUND(a.pct_high_income * 100.0, 2) AS pct_high_income_percent,
  ROUND(a.avg_age, 1) AS avg_age,
  ROUND(a.avg_hours, 2) AS avg_hours,
  t.occupation AS top_occupation
FROM agg a
LEFT JOIN top_occ t
  ON a.education = t.education AND a.sex = t.sex
ORDER BY a.pct_high_income DESC, a.total_people DESC
LIMIT 100;
```

## Why This Should Work Better

1. **Examples Match Pattern**: Complex examples show the exact pattern your query needs
2. **Explicit Guidance**: Prompts explicitly tell the model to use CTEs and ROW_NUMBER()
3. **Pattern Recognition**: Shows the model HOW to structure multi-step queries
4. **Multiple Techniques**: All 4 techniques now guide toward correct structure

## Testing

Test with your query:
1. Run: `streamlit run app.py`
2. Enter your question
3. Paste your gold SQL
4. Enable Evaluation Mode
5. Check if EM/EX/SM scores improve

The prompts should now generate SQL closer to your gold query structure!


