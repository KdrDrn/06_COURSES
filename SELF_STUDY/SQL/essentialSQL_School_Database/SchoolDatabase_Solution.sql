
--- To view ALL THE TABLES in the LibraryDB, execute the following QL DDL script

USE School
GO
SELECT * FROM INFORMATION_SCHEMA.TABLES 

--- To see ALL THE COLUMNS in the Books table, run the following script

SELECT COLUMN_NAME, DATA_TYPE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME='Department'

SELECT * FROM Department
SELECT * FROM Person
SELECT * FROM OnsiteCourse
SELECT * FROM OnlineCourse
SELECT * FROM StudentGrade
SELECT * FROM CourseInstructor
SELECT * FROM Course
SELECT * FROM OfficeAssignment
SELECT * FROM sysdiagrams



----- Problem 1
-- The schools would like to know which student has the highest GPA. To qualify students, need to have taken two or more courses.
-- List the Student Name, number of courses taken, and overall GPA.

SELECT COLUMN_NAME, DATA_TYPE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME='StudentGrade'

SELECT COLUMN_NAME, DATA_TYPE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME='Person'



SELECT 
	TOP 1 Person.FirstName + ' ' + Person.LastName AS StudentName,
	COUNT(StudentID) AS CoursesTaken,
	AVG(Grade) AS OverallGPA
FROM StudentGrade
INNER JOIN Person 
	ON StudentGrade.StudentID = Person.PersonID
GROUP BY Person.FirstName + ' ' + Person.LastName
HAVING COUNT(StudentID) >= 2
ORDER BY OverallGPA DESC;	



----- Problem 2
-- In order to plan for some new hires, the school would like to know which course is most popular.
-- Create a cross tab counting the number of courses taken. The row should be student enrollment year, the columns, course title.

SELECT COLUMN_NAME, DATA_TYPE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME='Person'

SELECT COLUMN_NAME, DATA_TYPE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME='StudentGrade'

SELECT COLUMN_NAME, DATA_TYPE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME='Course'



SELECT 
	EnrollmentYear,
	Calculus,
	Chemistry,
	Composition,
	Literature,
	Macroeconomics,
	Microeconomics,
	Physics,
	Poetry,
	Quantitative,
	Trigonometry
FROM(
	SELECT 
		YEAR(P.EnrollmentDate) EnrollmentYear,
		SG.EnrollmentID,
		C.Title
	FROM Person P
		INNER JOIN StudentGrade SG ON P.PersonID = SG.StudentID
		INNER JOIN Course C ON SG.CourseID = C.CourseID
	) AS PivotData 
PIVOT(
	COUNT(EnrollmentID) 
	FOR Title IN(Calculus,Chemistry,Composition,Literature,Macroeconomics,Microeconomics,Physics,Poetry,Quantitative,Trigonometry)
	) AS PivotResult
ORDER BY EnrollmentYear;



----- Problem 3
-- The facilities manager would like to create a directory of Instructor Offices. 
-- To help them out, you’re going to provide them a result containing the Building Name, Office Number, and Instructor Name.
-- The results should be ordered by Building Name and then Office Number.

SELECT COLUMN_NAME, DATA_TYPE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME='OfficeAssignment'

SELECT COLUMN_NAME, DATA_TYPE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME='Person'

SELECT 
	SUBSTRING(OA.Location, CHARINDEX(' ', OA.Location) + 1, LEN(OA.Location) - CHARINDEX(' ', OA.Location)) AS Building,
	LEFT(OA.Location, CHARINDEX(' ', OA.Location) - 1) AS OfficeNumber,
	P.FirstName + ' ' + P.LastName AS InstructorName
FROM 
	OfficeAssignment OA
INNER JOIN 
	Person P 
ON P.PersonID = OA.InstructorID
WHERE P.Discriminator = 'Instructor'
ORDER BY Building,OfficeNumber;



----- Problem 4
-- There is a fear the Person table, which isn’t normalized, has a data issue.
-- The column discriminator should either be “Instructor” or “Student” depending on whether HireDate or EnrollmentDate, respectively are filled.
-- Write a query to check the discriminator colum is correctly based on HireDate and EnrollmentDate. Your query should output the PersonID,
-- LastName, DatePresent (Hire or Enrollment), Flag. The Flag will be “Discriminator OK” if the data check out, and “Discriminator Bad” if it doesn’t.

SELECT 
	PersonID,
	LastName,
	COALESCE(HireDate, EnrollmentDate) AS DatePresent,
	HireDate,
	EnrollmentDate,
	Discriminator,
	CASE
		WHEN HireDate IS NOT NULL AND EnrollmentDate IS NOT NULL THEN 'Discriminator Bad'
		WHEN HireDate IS NULL AND EnrollmentDate IS NULL THEN 'Discriminator Bad'
		WHEN HireDate IS NOT NULL AND Discriminator = 'Instructor' THEN 'Discriminator OK'
		WHEN EnrollmentDate IS NOT NULL AND Discriminator = 'Student' THEN 'Discriminator OK'
	ELSE 'Discriminator Bad'
	END AS Flag
FROM Person
ORDER BY PersonID;



----- Problem 5
-- Create course schedules for all student’s onsite courses Include the Student Name, Department Name, Course Title, Location, Days and Time.
-- Order the schedule by student last name, department name, and course id. 

SELECT 
	S.FirstName + ' ' + S.LastName AS [StudentName],
	D.Name AS [DepartmentName],
	C.Title AS [Course Title],
	OC.Location,
	OC.Days,
	OC.Time
FROM
	Person S
INNER JOIN StudentGrade SG 
	ON StudentID = S.PersonID
INNER JOIN Course C 
	ON C.CourseID = SG.CourseID
INNER JOIN Department D 
	ON D.DepartmentID = C.DepartmentID
INNER JOIN OnSiteCourse OC 
	ON OC.CourseID = C.CourseID
ORDER BY 
	S.LastName,
	D.Name,
	C.CourseID;






