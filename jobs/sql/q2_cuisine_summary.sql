-- Q2 (join + aggregation): rating and share of 1-2 star reviews per cuisine
SELECT r.cuisine,
       COUNT(DISTINCT r.business_id) AS restaurants,
       COUNT(*) AS n_reviews,
       ROUND(AVG(v.stars), 3) AS avg_stars,
       ROUND(SUM(CASE WHEN v.stars <= 2 THEN 1 ELSE 0 END) / COUNT(*), 4) AS neg_share
FROM reviews v
JOIN restaurants r ON v.business_id = r.business_id
GROUP BY r.cuisine
ORDER BY avg_stars DESC, cuisine
