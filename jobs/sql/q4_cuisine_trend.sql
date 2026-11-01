-- Q4 (aggregation + LAG window): yearly average rating per cuisine and change from the previous year
-- {FIRST_YEAR} and {LAST_FULL_YEAR} are filled in by 02_sql_analytics.py: the last six full years in the data
SELECT y.cuisine,
       y.review_year,
       y.n AS n_reviews,
       ROUND(y.a, 3) AS avg_stars,
       ROUND(y.a - LAG(y.a) OVER (PARTITION BY y.cuisine ORDER BY y.review_year), 3) AS yoy_change
FROM (
    SELECT r.cuisine, v.review_year, COUNT(*) AS n, AVG(v.stars) AS a
    FROM reviews v
    JOIN restaurants r ON v.business_id = r.business_id
    WHERE v.review_year BETWEEN {FIRST_YEAR} AND {LAST_FULL_YEAR}
    GROUP BY r.cuisine, v.review_year
) y
ORDER BY cuisine, review_year
