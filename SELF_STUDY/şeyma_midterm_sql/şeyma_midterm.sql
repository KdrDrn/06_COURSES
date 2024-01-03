CREATE database seyma_midterm

USE seyma_midterm

CREATE TABLE customers (customer_id INT IDENTITY(101,1) CONSTRAINT customer_id_pk PRIMARY KEY, 
                        last_name VARCHAR(25) NOT NULL, 
						first_name VARCHAR(25) NOT NULL, 
						home_phone VARCHAR(12) NOT NULL, 
						address VARCHAR(100) NOT NULL, 
						city VARCHAR(30) NOT NULL, 
						state VARCHAR(2) NOT NULL, 
						email VARCHAR(25), 
						cell_phone VARCHAR(12) );

SELECT * FROM customers;

CREATE TABLE movies (title_id INT IDENTITY(1,1) CONSTRAINT title_id_pk PRIMARY KEY, 
                     title VARCHAR(60) NOT NULL, 
					 description VARCHAR(400) NOT NULL, 
					 rating VARCHAR(4) CONSTRAINT movies_rating CHECK (rating IN ('G', 'PG','R','PG13')), 
					 category VARCHAR(20) CHECK (category IN ('DRAMA','COMEDY', 'ACTION', 'CHILD', 'SCIFI', 'DOCUMENTARY')), 
					 release_date date NOT NULL);

SELECT * FROM movies;

CREATE TABLE media (media_id INT IDENTITY(92,1) CONSTRAINT media_id_pk PRIMARY KEY, 
                   format VARCHAR(3) NOT NULL, 
				   title_id INT NOT NULL CONSTRAINT media_titleid_fk REFERENCES movies(title_id)); 

SELECT * FROM media;

CREATE TABLE rental_history (media_id INT CONSTRAINT media_id_fk REFERENCES media(media_id), 
                             rental_date date DEFAULT GETDATE() NOT NULL, 
							 customer_id INT NOT NULL CONSTRAINT customer_id_fk REFERENCES customers(customer_id), 
							 return_date date, CONSTRAINT rental_history_pk PRIMARY KEY (media_id, rental_date));

SELECT * FROM rental_history;

CREATE TABLE actors (actor_id INT IDENTITY(1001,1) CONSTRAINT actor_id_pk PRIMARY KEY, 
                     stage_name VARCHAR(40) NOT NULL,
					 first_name VARCHAR(25) NOT NULL,
					 last_name VARCHAR(25) NOT NULL,
					 birth_date date NOT NULL);

SELECT * FROM actors;

CREATE TABLE roles (actor_id INT CONSTRAINT actor_id_fk REFERENCES actors(actor_id), 
             title_id INT CONSTRAINT title_id_fk REFERENCES movies(title_id), 
			 comments VARCHAR(40), CONSTRAINT star_billings_pk PRIMARY KEY (actor_id, title_id)); 

SELECT * FROM roles;



INSERT INTO customers (last_name, first_name, home_phone, address, city, state, email, cell_phone)
VALUES 
    ('Palombo', 'Lisa', '716-270-2669', '123 Main St', 'Buffalo', 'NY', 'palombo@ecc.edu', '716-555-1222'),
    ('Durak', 'Þeyma', '262-646-1111', '111 Akse St', 'Gaziler', 'KO', 'seyma@example.com', '532-542-5521'),
	('Durak', 'Sena', '262-646-1112', '111 Akse St', 'Gaziler', 'KO', 'sena@example.com', '532-542-5522'),
	('Durak', 'Yahya', '262-646-1113', '111 Akse St', 'Gaziler', 'KO', 'yahya@example.com', '532-542-5523'),
	('Durak', 'Mustafa', '262-646-1114', '111 Akse St', 'Gaziler', 'KO', 'mustafa@example.com', '532-542-5524'),
	('Durak', 'Hatice', '262-646-1115', '111 Akse St', 'Gaziler', 'KO', 'hatice@example.com', '532-542-5525'),
	('Durak', 'Sevgi', '262-646-1116', '111 Akse St', 'Gaziler', 'KO', 'sevgi@example.com', '532-542-5526');

SELECT * FROM customers;

INSERT INTO movies (title, description, rating, category, release_date) 
VALUES ('Remember the Titans','Story of coach and team', 'PG','DRAMA','29- SEP-2000'),
       ('Killers of the Flower Moon','When oil is discovered in 1920s Oklahoma under Osage Nation land, the Osage people are murdered one by one - until the FBI steps in to unravel the mystery.', 'R','ACTION','19- DEC-2006'),
	   ('Dunki','Four friends from a village in Punjab share a common dream: to go to England. Their problem is that they have neither the visa nor the ticket. A soldier promises to take them to the land of their dreams.', 'G','COMEDY','29- NOV-2021'),
	   ('Aquaman and the Lost Kingdom','Black Manta seeks revenge on Aquaman for his father death. To defend Atlantis, Aquaman forges an alliance with his imprisoned brother. They must protect the kingdom.', 'PG13','DOCUMENTARY','14- MAY-2019'),
	   ('7th Heaven','Eric Camden, a minister, and his wife Annie deal with the drama of having seven children, ranging from toddlers to adults with families of their own.', 'PG','DRAMA','22- APR-2012'),
	   ('Cry Macho','A one-time rodeo star and washed-up horse breeder takes a job to bring a man young son home and away from his alcoholic mom. On their journey, the horseman finds redemption through teaching the boy what it means to be a good man.', 'PG','SCIFI','19-MAR-2005'),
	   ('Death Warrant','In a violent and corrupt prison, decorated cop Louis Burke must infiltrate the jail to find answers to a number of inside murders. What he finds is a struggle of life and death tied in to his own past.', 'R','CHILD','29- SEP-2000');

SELECT * FROM movies;

INSERT INTO media (format, title_id) 
VALUES ('DVD',1),
       ('CD',1),
       ('BLU',1),
	   ('VHS',1),
       ('DVD',2),
       ('CD',2),
       ('BLU',2),
	   ('VHS',2),
       ('DVD',3),
       ('CD',3),
       ('BLU',3),
	   ('VHS',3),
       ('DVD',4),
       ('CD',4),
       ('BLU',4),
	   ('VHS',4),
       ('DVD',5),
       ('CD',5),
       ('BLU',5),
	   ('VHS',5),
       ('DVD',6),
       ('CD',6),
       ('BLU',6),
	   ('VHS',6),
       ('DVD',7),
       ('CD',7),
       ('BLU',7),
	   ('VHS',7);

SELECT * FROM media;

INSERT INTO rental_history (media_id, rental_date, customer_id, return_date)
VALUES
    (120, '19-SEP-2010', 101, '26-SEP-2010'),
	(121, '22-DEC-2017', 105, '12-FEB-2018'),
    (122, GETDATE(), 101, NULL),
    (123, GETDATE(), 102, NULL),
    (124, GETDATE(), 103, NULL),
    (125, GETDATE(), 104, NULL),
    (126, GETDATE(), 105, NULL);

SELECT * FROM rental_history;

INSERT INTO actors (stage_name, first_name, last_name, birth_date)
VALUES 
    ('Dog Fight', 'Bradt', 'Pitt', '18-DEC-1963'),
    ('The Magnificent', 'Tom', 'Hanks', '09-JUL-1956'),
    ('Dazzling Diva', 'Angelina', 'Jolie', '04-JUN-1975'),
    ('Charming Star', 'Chris', 'Hemsworth', '11-AUG-1983'),
    ('Versatile Virtuoso', 'Meryl', 'Streep', '12-JUN-1949'),
    ('Dynamic Dynamo', 'Idris', 'Elba', '06-SEP-1972');

SELECT * FROM actors;

INSERT INTO roles (actor_id, title_id, comments)
VALUES 
    (1002, 1, 'Romantic Lead'),
    (1003, 2, 'Perfect'),
    (1004, 3, 'Exciting'),
    (1005, 4, 'Normal'),
    (1006, 5, 'Sadly'),
    (1007, 6, 'Disappointment');

SELECT * FROM roles;

INSERT INTO rental_history (media_id, rental_date, customer_id, return_date)
VALUES
    (129, '19-SEP-2010', 102, '26-SEP-2010'),
	(128, '22-DEC-2017', 103, '12-FEB-2018');

SELECT * FROM rental_history;

SELECT * 
FROM customers
ORDER BY customer_id ASC;

SELECT * 
FROM movies
ORDER BY title_id ASC;

SELECT * 
FROM media
ORDER BY media_id ASC;

SELECT * 
FROM rental_history
ORDER BY rental_date ASC;

SELECT * 
FROM actors
ORDER BY actor_id ASC;

SELECT * 
FROM roles
ORDER BY actor_id ASC;


CREATE VIEW NOT_YET_RETURNED AS

SELECT *
FROM rental_history
WHERE return_date IS NULL;

SELECT * FROM NOT_YET_RETURNED;