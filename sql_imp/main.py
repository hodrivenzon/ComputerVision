import sqlite3
from sqlite3 import Error


def create_connection(db_file):
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)

    return None


def create_table(conn, create_table_sql):
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)


def select_query(conn, query):
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    for row in rows:
        print(row)


def insert_query(conn, query):
    cur = conn.cursor()
    cur.execute(query)


def delete_query(conn, query):
    cur = conn.cursor()
    cur.execute(query)


def load_data(conn):
    sql_load_queries = """
            Delete from Doctor;
            INSERT INTO Doctor VALUES('111111111','DC','Dror','Cohen','Phd',2007,'Tel-Aviv');
            INSERT INTO Doctor VALUES('222222222','RM','Shimrit','Dabush','MA',2005,'Petah-Tikwa');
            INSERT INTO Doctor VALUES('333333333','DH','Tania','Reznikov','Msc',2010,'Herzelia');
            INSERT INTO Doctor VALUES('444444444','DB','Ayala','BenZvi','Phd',2009,'Tel-Aviv');

            Delete from Doctor_Specialization;
            INSERT INTO Doctor_Specialization VALUES('111111111','Computer');
            INSERT INTO Doctor_Specialization VALUES('111111111','Systems Engineering');
            INSERT INTO Doctor_Specialization VALUES('222222222','Systems Engineering');
            INSERT INTO Doctor_Specialization VALUES('444444444','Artificial Intelligence');
            INSERT INTO Doctor_Specialization VALUES('444444444','DataBase');
            INSERT INTO Doctor_Specialization VALUES('111111111','Engineering');
            INSERT INTO Doctor_Specialization VALUES('333333333','Engineering');


            Delete from Patients;
            INSERT INTO Patients VALUES('777777777','PP','Dor','Matazfi','03/10/2001','bla','Rona','David');
            INSERT INTO Patients VALUES('888888888','CC','Dror','Samet','23/05/1991','blabla','Tiki','Dani');
            INSERT INTO Patients VALUES('787878787','SS','Galit','Naim','12/07/1997','blab','Ruth','Moshe');
            INSERT INTO Patients VALUES('999999999','EE','Guy','shani','15/11/1986','bbbb','Orna','Asher');
            INSERT INTO Patients VALUES('191919191','BB','Mordechai','BenDror','25/10/2000','bbbb','Tami','Oren');
            INSERT INTO Patients VALUES('232323232','EE','Shiran','Aliashev','08/12/1985','bbabb','Lea','Shlomi');

            Delete from Treatment;
            INSERT INTO Treatment VALUES(111,'Name1','Bla Bla Bla Bla');
            INSERT INTO Treatment VALUES(333,'Name2','Bla Bla');
            INSERT INTO Treatment VALUES(444,'Name3','Bla Bla Bla');
            INSERT INTO Treatment VALUES(555,'Name4','Bla');


            Delete from Shifts;
            INSERT INTO Shifts VALUES('111111111','01/02/2019','13:00','20:00');
            INSERT INTO Shifts VALUES('222222222','02/02/2019','13:00','20:00');
            INSERT INTO Shifts VALUES('444444444','01/02/2019','14:00','21:00');
            INSERT INTO Shifts VALUES('111111111','03/02/2019','09:00','16:00');
            INSERT INTO Shifts VALUES('333333333','01/02/2019','13:00','21:00');
            INSERT INTO Shifts VALUES('222222222','05/02/2019','13:00','20:00');

            Delete from Labs;
            INSERT INTO Labs VALUES(99,'NameLab1','bla');
            INSERT INTO Labs VALUES(88,'NameLab2','blabla');
            INSERT INTO Labs VALUES(11,'NameLab3','blablabal');
            INSERT INTO Labs VALUES(77,'NameLab4','bbbbbb');
            INSERT INTO Labs VALUES(33,'NameLab5','aaaa');
            INSERT INTO Labs VALUES(44,'NameLab6','blablabla');

            Delete from Patinets_Labs;
            INSERT INTO Patinets_Labs VALUES(99,'01/01/2019','777777777','Good');
            INSERT INTO Patinets_Labs VALUES(11,'01/01/2019','777777777','VeryGood');
            INSERT INTO Patinets_Labs VALUES(99,'12/12/2018','777777777','Bad');
            INSERT INTO Patinets_Labs VALUES(44,'24/05/2018','999999999','Good');
            INSERT INTO Patinets_Labs VALUES(33,'01/01/2019','999999999','VeryGood');
            INSERT INTO Patinets_Labs VALUES(99,'01/01/2019','787878787','Very Very Good');
            INSERT INTO Patinets_Labs VALUES(77,'12/12/2018','999999999','Good');
            INSERT INTO Patinets_Labs VALUES(11,'02/02/2018','888888888','Bad');
            INSERT INTO Patinets_Labs VALUES(33,'22/03/2019','191919191','Bad');


            Delete from Patinets_Treatments;
            INSERT INTO Patinets_Treatments VALUES('999999999',333,'333333333','07/05/2019','15/07/2019');
            INSERT INTO Patinets_Treatments VALUES('888888888',555,'333333333','07/11/2018','12/07/2019');
            INSERT INTO Patinets_Treatments VALUES('999999999',111,'333333333','07/05/2019','15/07/2019');
            INSERT INTO Patinets_Treatments VALUES('777777777',333,'111111111','17/05/2019','15/07/2019');
            INSERT INTO Patinets_Treatments VALUES('777777777',111,'444444444','27/06/2019','15/07/2019');
            INSERT INTO Patinets_Treatments VALUES('191919191',555,'444444444','01/01/2019','15/07/2019');
            INSERT INTO Patinets_Treatments VALUES('191919191',111,'222222222','15/05/2019','15/07/2019');
            INSERT INTO Patinets_Treatments VALUES('888888888',333,'222222222','01/01/2019','15/07/2019');
            INSERT INTO Patinets_Treatments VALUES('999999999',555,'222222222','07/05/2019','15/07/2019');
            INSERT INTO Patinets_Treatments VALUES('777777777',444,'444444444','27/07/2018','15/07/2019');
            INSERT INTO Patinets_Treatments VALUES('777777777',555,'444444444','27/08/2018','15/07/2019');


            Delete from Patinets_Progress;
            INSERT INTO Patinets_Progress
            VALUES('999999999',333,'333333333','bla','bla bla','22/08/2018');
            INSERT INTO Patinets_Progress
            VALUES('999999999',111,'333333333','blabla','bla','30/07/2018');
            INSERT INTO Patinets_Progress
            VALUES('777777777',333,'111111111','bla','bla','15/10/2017');
            INSERT INTO Patinets_Progress
            VALUES('777777777',111,'444444444','blablabla','bla','25/12/2018');
            INSERT INTO Patinets_Progress
            VALUES('999999999',555,'222222222','bla','bla','15/12/2018');
            INSERT INTO Patinets_Progress 
            VALUES('777777777',444,'444444444','bla bla','bla','13/11/2018');
            INSERT INTO Patinets_Progress
            VALUES('999999999',555,'444444444','bla','bla','30/07/2018');"""

    queries = sql_load_queries.split(";")
    for query in queries:
        insert_query(conn, query)


def create_all_tables(conn):
    # Example- Create Doctor Table#
    sql_create_doctors_table = """CREATE TABLE Doctor
                                    (
                                       ID			char(9)	primary key,
                                       Psw          varchar(20) not null,
                                       First_Name	varchar(50) not null,
                                       Last_Name	varchar(50) not null,
                                       Degree       varchar(20),
                                       P_Year       char(10),
                                       Address      varchar(50)
                                    );
                                    """

    sql_create_doctors_specialization = """    CREATE TABLE Doctor_Specialization
                                        (
                                           ID		  char(9) references Doctor(ID) ON UPDATE CASCADE ON DELETE CASCADE,
                                           Specialty  varchar(50) default 'General',
                                           CONSTRAINT tab_pk PRIMARY KEY (ID,Specialty)
                                        );"""

    sql_create_patients = """CREATE TABLE Patients
                                        (
                                           Patient_ID	char(9) primary key,
                                           Patient_psw  varchar(20) not null,
                                           First_Name	varchar(20) not null,
                                           Last_Name    varchar(20) not null,
                                           Birthday     char(10),
                                           Address      varchar(50),
                                           Fathers_name varchar(20),
                                           Mothers_name varchar(20)
                                        );"""

    sql_create_treatment = """CREATE TABLE Treatment	
                                        (
                                            Code          int  primary key,
                                            T_Name	      varchar(20) not null,
                                            T_Description varchar(100) not null
                                        );"""

    sql_create_shifts = """CREATE TABLE Shifts
                                        (
                                            Employee_id char(9) references Doctor(ID)
                                            ON UPDATE CASCADE ON DELETE CASCADE,
                                            Date char(10),
                                            Start_time char(10),
                                            Finish_time char(10),
                                            CONSTRAINT shifts_pk PRIMARY KEY (Employee_ID,Date,Start_time),
                                            CONSTRAINT ch_wk check (Start_time<Finish_time)
    );"""

    sql_create_labs = """CREATE TABLE Labs
                                        (
                                            Lab_ID	 int primary key,
                                            Lab_name varchar(30) not null,
                                            Lab_desc varchar(100)
                                        );"""

    sql_create_patients_labs = """CREATE TABLE Patinets_Labs
                                        (
                                            Lab_ID	 int references Labs(Lab_id)
                                            ON UPDATE CASCADE ON DELETE CASCADE,
                                            Lab_date char(10),
                                            Patient_id char(9) references Patients(Patient_id)
                                            ON UPDATE CASCADE ON DELETE CASCADE,
                                            Result varchar(50) default 'none',
                                            CONSTRAINT PL_pk PRIMARY KEY (Lab_ID,Lab_date,Patient_id)
                                        );"""

    sql_create_patintes_treatments = """CREATE TABLE Patinets_Treatments
                                        (
                                            Patient_ID	 char(9) references Patients(Patient_id)
                                            ON UPDATE CASCADE ON DELETE CASCADE,
                                            Treatment_code int references Treatment(Code)
                                            ON UPDATE CASCADE ON DELETE CASCADE,
                                            Doctor_id char(9) references Doctor(ID)
                                            ON UPDATE CASCADE ON DELETE CASCADE,
                                            Start_date char(10),
                                            Finish_date char(10),
                                            CONSTRAINT PT_pk PRIMARY KEY (Patient_ID,Treatment_code,Doctor_id)
                                        ); """

    sql_create_patintes_progress = """CREATE TABLE Patinets_Progress
                                        (
                                            Patient_ID	 char(9),
                                            Treatment_code int,
                                            Doctor_id char(9),
                                            Initial_description varchar(100) not null,
                                            Present_description varchar(100) not null,
                                            Date_Ending char(10),     
                                            CONSTRAINT PP_pk PRIMARY KEY (Patient_ID,Treatment_code,Doctor_id),
                                            CONSTRAINT fk_pk FOREIGN KEY (Patient_ID,Treatment_code,Doctor_id)
                                            references Patinets_Treatments(Patient_ID,Treatment_code,Doctor_id)
                                            ON UPDATE CASCADE ON DELETE CASCADE); """

    create_table(conn, sql_create_doctors_table)
    create_table(conn, sql_create_doctors_specialization)
    create_table(conn, sql_create_patients)
    create_table(conn, sql_create_treatment)
    create_table(conn, sql_create_shifts)
    create_table(conn, sql_create_labs)
    create_table(conn, sql_create_patients_labs)
    create_table(conn, sql_create_patintes_treatments)
    create_table(conn, sql_create_patintes_progress)
    # Your code in here- Create all other tables


def create_all_queries(conn):
    # Example- Q1#
    sql_q1_query = """select First_Name,Last_Name
                                    from Doctor,shifts
                                    where Doctor.ID=shifts.Employee_id and shifts.Date='01/02/2019'
                                    ; """
    select_query(conn, sql_q1_query)

    # Your code in here- Answer all other questions


def main():
    database = r"C:\Users\hodda\Downloads\DBexample.db"

    conn = create_connection(database)
    if conn is not None:
        create_all_tables(conn)
        load_data(conn)
        create_all_queries(conn)
    else:
        print("Error! cannot create the database connection.")


if __name__ == '__main__':
    main()
