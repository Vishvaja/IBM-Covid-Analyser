import sqlite3
from sqlite3 import Error




def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except Error as e:
        print(e)
    finally:
        if conn:
            conn.close()


if __name__ == '__main__':
    create_connection("data2.db")
    
conn = sqlite3.connect("data2.db")
print(sqlite3.version)
conn.execute('''CREATE TABLE COMPANY
         (ID INT PRIMARY KEY     NOT NULL,
         NAME           TEXT    NOT NULL,
         AGE            INT     NOT NULL,
         ADDRESS        CHAR(50),
         SALARY         REAL);''')
print ("Table created successfully");
conn.execute("INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY) \
      VALUES (2, 'Poaul', 32, 'Caliwfornia', 520000.00 )");
print ("Values inserted successfully");
conn.commit()

conn.close()    
    


import sqlite3

conn = sqlite3.connect('data2.db')
print ("Opened database successfully");
cur = conn.cursor()
var='''SELECT * FROM COMPANY'''
cur.execute(var)

rows = cur.fetchall()

for row in rows:
   print ("ID = ", row[0])
   print ("NAME = ", row[1])
   print ("ADDRESS = ", row[2])
   print ("SALARY = ", row[3], "\n")

print ("Operation done successfully");


conn.close()
