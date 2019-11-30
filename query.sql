	
SELECT a.product_id, b.product_id as copurchased, count(*) as times_bought_together
FROM public.order_details AS a
INNER JOIN public.order_details b ON a.order_id = b.order_id
AND a.product_id != b.product_id
GROUP BY a.product_id, b.product_id
order by a.product_id