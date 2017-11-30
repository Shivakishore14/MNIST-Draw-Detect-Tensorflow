import psycopg2
import os
import urlparse

class DataStore:
    def __init__(self):
        # DATABASE_URL = 'postgres://mnistapp:mnistapppass@localhost/mnistappdb'
        urlparse.uses_netloc.append("postgres")
        url = urlparse.urlparse(os.environ['DATABASE_URL'])

        conn = psycopg2.connect(
            database=url.path[1:],
            user=url.username,
            password=url.password,
            host=url.hostname,
            port=url.port
        )
        CREATE_TABLE = "CREATE TABLE IF NOT EXISTS key_val_store( key_ text , value_ int);"
        CREATE_ROW = "INSERT INTO key_val_store(key_, value_) SELECT '%s' , 0 WHERE '%s' NOT IN ( SELECT key_ FROM key_val_store);"
        keys = ["visits", "uniq_visits", "prediction_reqs"]

        cur = conn.cursor()
        cur.execute(CREATE_TABLE)
        for k in keys:
            cur.execute(CREATE_ROW%(k,k))

        self.cur = cur
        self.conn = conn
        self.commit()

    def commit(self):
        self.conn.commit()

    def get_data_from_db(self):
        key_val = {}
        try:
            self.cur.execute("""SELECT * from key_val_store""")
        except:
            print("Error inserting to table")

        rows = self.cur.fetchall()
        for row in rows:
            key_val[row[0]] = row[1]
        return key_val
        self.commit()

    def update_key(self, key):
        try:
            query = "UPDATE key_val_store SET value_ = value_ + 1 WHERE key_ = '%s'"%(key)
            self.cur.execute(query)
        except:
            print("Update Error")
        self.commit()
