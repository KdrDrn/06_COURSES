--- sunucu:94.73.146.4
--- ka:u0583688_cinar
--- pass:Semih1234566Cinar.


select * from dbo.Query1



 select MalKodu, sum(Miktar) as NetStok
 from dbo.Query1
 group by MalKodu
 order by MalKodu;

Truncate table dbo.stok

SELECT
		MalKodu,
		SUM(CASE WHEN IslemTur = 1 THEN 1*Miktar END) AS Çýkýþ,
		SUM(CASE WHEN IslemTur = 0 THEN 1*Miktar END) AS Stok
INTO stok
FROM dbo.Query1
GROUP BY MalKodu

SELECT MalKodu, Stok, Çýkýþ, COALESCE(Stok, 0) - COALESCE(Çýkýþ, 0) AS NetStok
FROM dbo.stok;



--- SEMÝH HOCA
select sub0.MalKodu, sub0.Miktar - sub1.Miktar as stock
from
(
SELECT        SUM(Miktar) AS Miktar, IslemTur, MalKodu
FROM            Query1
where MalKodu = '10081 SÝEMENS' and IslemTur = 0
GROUP BY IslemTur, MalKodu
)sub0
,
(
SELECT        SUM(Miktar) AS Miktar, IslemTur, MalKodu
FROM            Query1
where MalKodu = '10081 SÝEMENS' and IslemTur = 1
GROUP BY IslemTur, MalKodu
)sub1




--- BENJAMIN HOCA
SELECT MalKodu,
SUM(CASE WHEN IslemTur = 0 THEN Miktar ELSE -Miktar END) Miktar     
FROM  Query1
WHERE IslemTur IN('1', '0')
GROUP BY MalKodu
order by MalKodu