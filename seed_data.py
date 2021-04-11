import sqlite3

connection = sqlite3.connect('student_courses.db')

cursor = connection.cursor()

command1 = """CREATE TABLE IF NOT EXISTS 
student_courses(student_id INTEGER PRIMARY KEY, courses TEXT)"""

cursor.execute(command1)

cursor.execute("INSERT INTO student_courses VALUES (17090,'Course A, Course B')")
cursor.execute("INSERT INTO student_courses VALUES (17091,'Course C, Course D')")
cursor.execute("INSERT INTO student_courses VALUES (17092,'Course E, Course F')")

cursor.execute("SELECT * FROM student_courses")

results = cursor.fetchall()
print(results)

cursor.execute("SELECT courses FROM student_courses WHERE student_id=17090")

results = cursor.fetchall()
print(results[0][0])

