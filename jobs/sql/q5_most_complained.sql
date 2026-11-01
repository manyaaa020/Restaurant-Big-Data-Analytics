-- Q5 (join + HAVING + sort): restaurants with the highest share of 1-2 star reviews (min 200 reviews)
SELECT r.business_id,
       r.name,
       r.city,
       r.cuisine,
       COUNT(*) AS n_reviews,
       ROUND(AVG(v.stars), 3) AS avg_stars,
       ROUND(SUM(CASE WHEN v.stars <= 2 THEN 1 ELSE 0 END) / COUNT(*), 4) AS neg_share
FROM reviews v
JOIN restaurants r ON v.business_id = r.business_id
GROUP BY r.business_id, r.name, r.city, r.cuisine
HAVING COUNT(*) >= 200
ORDER BY neg_share DESC, n_reviews DESC, business_id
LIMIT 25
