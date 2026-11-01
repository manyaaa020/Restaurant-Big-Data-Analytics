-- Q3 (nested subqueries + join + window ranking): rank cuisines by average rating within the 8 busiest cities
SELECT cc.city,
       cc.cuisine,
       cc.n AS n_reviews,
       ROUND(cc.avg_stars, 3) AS avg_stars,
       RANK() OVER (PARTITION BY cc.city ORDER BY cc.avg_stars DESC) AS rnk
FROM (
    SELECT r.city, r.cuisine, COUNT(*) AS n, AVG(v.stars) AS avg_stars
    FROM reviews v
    JOIN restaurants r ON v.business_id = r.business_id
    WHERE r.cuisine <> 'Other'
    GROUP BY r.city, r.cuisine
    HAVING COUNT(*) >= 300
) cc
JOIN (
    SELECT r2.city, COUNT(*) AS total
    FROM reviews v2
    JOIN restaurants r2 ON v2.business_id = r2.business_id
    GROUP BY r2.city
    ORDER BY total DESC, city
    LIMIT 8
) tc ON cc.city = tc.city
ORDER BY city, rnk, cuisine
