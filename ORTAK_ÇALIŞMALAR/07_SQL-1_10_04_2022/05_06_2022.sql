-- database test_4

select * from [dbo].[Calisan_Ana_Tablo]

select * from [dbo].[Calisan_Online_Kontrol]

-- create d�zenler
-- alter de�i�tirir
-- 

-- trigger create ediyoruz
-- syntax a�a��daki gibi
create trigger trigger_calisan_kontrol
on [dbo].[Calisan_Ana_Tablo]
after insert
as
begin

insert into [dbo].[Calisan_Online_Kontrol] select cardID, GirisCikis, Tarih from inserted

end

-- �al��an ana tabloya veri giri�i yapt�k
insert into [dbo].[Calisan_Ana_Tablo] values ('kadir', 'duran', 'giri�', getdate(), 101)

-- her 2 tablodaki kay�tlar� kontrol ediyoruz
select * from [dbo].[Calisan_Ana_Tablo]
select * from [dbo].[Calisan_Online_Kontrol]

-- �al��an ana tabloya veri giri�i yapt�k
insert into [dbo].[Calisan_Ana_Tablo] values ('sam', 'steady', 'giri�', getdate(), 102)

-- her 2 tablodaki kay�tlar� kontrol ediyoruz
select * from [dbo].[Calisan_Ana_Tablo]
select * from [dbo].[Calisan_Online_Kontrol]

-- �al��an ana tabloya veri giri�i yapt�k
insert into [dbo].[Calisan_Ana_Tablo] values ('ken', 'durans', 'giri�', getdate(), 103)

-- her 2 tablodaki kay�tlar� kontrol ediyoruz
select * from [dbo].[Calisan_Ana_Tablo]
select * from [dbo].[Calisan_Online_Kontrol]

-- �al��an ana tabloya veri giri�i yapt�k
insert into [dbo].[Calisan_Ana_Tablo] values ('sam', 'steady', '�iki�', getdate(), 102)

-- her 2 tablodaki kay�tlar� kontrol ediyoruz
select * from [dbo].[Calisan_Ana_Tablo]
select * from [dbo].[Calisan_Online_Kontrol]





CREATE TABLE [dbo].[Trigger_stok_online](	[id] [int] IDENTITY(1,1) NOT NULL,	[urun_id] [int] NULL,	[stock] [int] NULL,) ON [PRIMARY]  CREATE TABLE [dbo].[Trigger_Urun](	[id] [int] IDENTITY(1,1) NOT NULL,	[urun_id] [int] NULL,	[stock] [int] NULL,	[girisCikis] [int] NULL,	[tarih] [date] NULL) ON [PRIMARY]


CREATE Trigger [dbo].[StokGuncelle]On [dbo].[Trigger_Urun]AFTER INSERTasBEGINdeclare @urun_id as int  -- urun_id de�i�kenini tan�ml�yorum. tan�mlamay� ba��una @ koyarak declare ile yap�yorumdeclare @stock as intdeclare @girisCikis as intdeclare @tarih as dateselect @urun_id = urun_id, @stock = stock, @girisCikis = girisCikis , @tarih = tarih  FROM inserted-- inserted i�indeki urun_id yi @urun_id de�i�kenime at�yorum.-- bu bana yap�lan son i�lemi bir de�ikene almam� sa��l�yor--IF @girisCikis = 1	--				update dbo.Trigger_stok_online		--			set stock = stock + @stock			--		where urun_id = @urun_id--IF @girisCikis = 2--					update dbo.Trigger_stok_online	--				set stock = stock - @stock		--			where urun_id = @urun_id		--insert into dbo.Trigger_stok_online select @urun_id, @stock from insertedIF @girisCikis = 1		update  dbo.Trigger_stok_online set stock = stock + @stock where urun_id = @urun_idIF @girisCikis = 2		update  dbo.Trigger_stok_online set stock = stock - @stock where urun_id = @urun_idIF @urun_id NOT IN (select urun_id from Trigger_stok_online)		insert into Trigger_stok_online select @urun_id,  @stock from insertedEND


select * from stok
select * from Trigger_stok_online
select * from Trigger_Urun


insert into [dbo].[Trigger_Urun] ([urun_id], [stock], [girisCikis], [tarih])
values (1,10,1,getdate())


select * from stok
select * from Trigger_stok_online
select * from Trigger_Urun


insert into [dbo].[Trigger_Urun] ([urun_id], [stock], [girisCikis], [tarih])
values (2,10,1,getdate())


select * from stok
select * from Trigger_stok_online
select * from Trigger_Urun


insert into [dbo].[Trigger_Urun] ([urun_id], [stock], [girisCikis], [tarih])
values (2,20,1,getdate())


select * from stok
select * from Trigger_stok_online
select * from Trigger_Urun


insert into [dbo].[Trigger_Urun] ([urun_id], [stock], [girisCikis], [tarih])
values (2,2,2,getdate())


select * from stok
select * from Trigger_stok_online
select * from Trigger_Urun



insert into [dbo].[Trigger_Urun] ([urun_id], [stock], [girisCikis], [tarih])
values (3,70,1,getdate())


select * from stok
select * from Trigger_stok_online
select * from Trigger_Urun




insert into [dbo].[Trigger_Urun] ([urun_id], [stock], [girisCikis], [tarih])
values (4,100,1,getdate())

select * from stok
select * from Trigger_stok_online
select * from Trigger_Urun




insert into [dbo].[Trigger_Urun] ([urun_id], [stock], [girisCikis], [tarih])
values (3,100,2,getdate())

select * from stok
select * from Trigger_stok_online
select * from Trigger_Urun