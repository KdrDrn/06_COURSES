

--create database...
DROP DATABASE IF EXISTS Zomato;
CREATE DATABASE Zomato;

drop table if exists goldusers_signup;
CREATE TABLE goldusers_signup(userid integer,gold_signup_date date); 

INSERT INTO goldusers_signup(userid,gold_signup_date) 
 VALUES (1,'09-22-2017'),
(3,'04-21-2017');

drop table if exists users;
CREATE TABLE users(userid integer,signup_date date); 

INSERT INTO users(userid,signup_date) 
 VALUES (1,'09-02-2014'),
(2,'01-15-2015'),
(3,'04-11-2014');

drop table if exists sales;
CREATE TABLE sales(userid integer,created_date date,product_id integer); 

INSERT INTO sales(userid,created_date,product_id) 
 VALUES (1,'04-19-2017',2),
(3,'12-18-2019',1),
(2,'07-20-2020',3),
(1,'10-23-2019',2),
(1,'03-19-2018',3),
(3,'12-20-2016',2),
(1,'11-09-2016',1),
(1,'05-20-2016',3),
(2,'09-24-2017',1),
(1,'03-11-2017',2),
(1,'03-11-2016',1),
(3,'11-10-2016',1),
(3,'12-07-2017',2),
(3,'12-15-2016',2),
(2,'11-08-2017',2),
(2,'09-10-2018',3);


drop table if exists product;
CREATE TABLE product(product_id integer,product_name text,price integer); 

INSERT INTO product(product_id,product_name,price) 
 VALUES
(1,'p1',980),
(2,'p2',870),
(3,'p3',330);


select * from sales;
select * from product;
select * from goldusers_signup;
select * from users;



1 ---- what is total amount each customer spent on zomato ?



2 ---- How many days has each customer visited zomato?



3 --- what was the first product purchased by each customer?




4 --- what is most purchased item on menu & how many times was it purchased by all customers ?



5 ---- which item was most popular for each customer?



6 --- which item was purchased first by customer after they become a member ?



7 --- which item was purchased just before customer became a member?



8 ---- what is total orders and amount spent for each member before they become a member ?



9 --- if buying each product generate points for eg 5rs=2 zomato point and each product has different purchasing points 
-- for eg for p1 5rs=1 zomato point,for p2 10rs=zomato point and p3 5rs=1 zomato point  2rs =1zomato point
,--calculate points collected by each customers and for which product most points have been given till now.



10 --- in the first one year after customer joins the gold program (including the join date ) irrespective of 
  --  what customer has purchased earn 5 zomato points for every 10rs spent who earned more more 1 or 3
   -- what int earning in first yr ? 1zp = 2rs



11 --- rnk all transaction of the customers



12 --- rank all transaction for each member whenever they are zomato gold member for every non gold member transaction mark as na

