import mysql.connector
"""
basic database
"""
# data science from sanjay
conn = mysql.connector.connect(
    host="localhost",
    port=3306,
    user="root",
    password="Data7Dollar$",
    database="mysql_learner"
)
cursor_obj = conn.cursor()
statement = """select E.F_NAME,E.L_NAME, J.JOB_TITLE from employees E , jobs J where E.JOB_ID = J.JOB_IDENT and E.SEX = "F"  """
cursor_obj.execute(statement)
print("All the data")
output = cursor_obj.fetchall()
for row in output:
    print(row)


cursor_obj.close()
conn.close()
