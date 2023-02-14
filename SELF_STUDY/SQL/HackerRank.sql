-- CREATING DATABASE

CREATE DATABASE HACKERRANK;

-- CREATING TABLE

CREATE TABLE OCCUPATIONS	(
							Name varchar(255),
							Occupation varchar(255)
							);

-- INSERT VALUES INTO TABLE

INSERT INTO HACKERRANK.dbo.OCCUPATIONS (Name, Occupation)

VALUES

('Samantha','Doctor'),
('Julia','Actor'),
('Maria','Actor'),
('Meera','Singer'),
('Ashely','Professor'),
('Ketty','Professor'),
('Christeen','Professor'),
('Jane','Actor'),
('Jenny','Doctor'),
('Priya','Singer')

SELECT * FROM OCCUPATIONS

SELECT
    Doctor,
    Actor,
    Singer,
    Professor
FROM OCCUPATIONS
PIVOT(
    MAX(Name)
    FOR Occupation IN ([Doctor],[Actor],[Singer],[Professor])) AS pvt

WITH pivot_data AS
(
SELECT
	Occupation, -- Spreading Column,
	Name -- Aggregate Column
FROM OCCUPATIONS 
)
SELECT [Doctor],[Actor],[Singer],[Professor]
FROM pivot_data
PIVOT 
(max(Name) 
FOR Occupation IN ([Doctor],[Actor],[Singer],[Professor])) AS pvt


  Select
     Min(Case Occupation When 'Doctor' Then Name End) Doctor,
     Min(Case Occupation When 'Actor' Then Name End) Actor,
     Min(Case Occupation When 'Singer' Then Name End) Singer,
     Min(Case Occupation When 'Professor' Then Name End) Professor
   From OCCUPATIONS


select
    Doctor,
    Professor,
    Singer,
    Actor
from (
    select
        NameOrder,
        max(case Occupation when 'Doctor' then Name end) as Doctor,
        max(case Occupation when 'Professor' then Name end) as Professor,
        max(case Occupation when 'Singer' then Name end) as Singer,
        max(case Occupation when 'Actor' then Name end) as Actor
    from (
            select
                Occupation,
                Name,
                row_number() over(partition by Occupation order by Name ASC) as NameOrder
            from OCCUPATIONS
         ) as NameLists
    group by NameOrder
    ) as Names


SELECT Doctor, Professor, Singer, Actor 
FROM (
	SELECT 
		ROW_NUMBER() OVER (PARTITION BY Occupation ORDER BY Name) AS rn, 
		Name, 
		Occupation 
	FROM OCCUPATIONS
	) AS A
PIVOT 
	(MAX(name) 
	FOR occupation 
	IN ([Doctor],[Actor],[Singer],[Professor])) AS PVT
ORDER BY rn;