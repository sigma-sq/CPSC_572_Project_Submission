import sqlalchemy as sql
import pandas as pd


class Storage:
    """
    Class representing a storage object. Use for convince getting data to and from database.

    Attributes:
        query (str): The SQL query to be executed.
        db (str): The name of the database.
        conn (object): The connection object to the database.
        cursor (object): The cursor object to execute SQL queries.
        engine (object): The SQL Alchemy engine object.

    Methods:
        connect(self)
            Establishes a connection to the database.

        save_data(self, name, data, replace=False)
            Saves the data to the specified table in the database.
            Args:
                name (str): The name of the table.
                data (pandas.DataFrame): The data to be saved.
                replace (bool, optional): If True, the existing data in the table will be replaced.
                                          If False, the data will be appended. Defaults to False.
            Returns:
                int: 0 if the data is saved successfully.

        export_table(self, table)
            Exports the specified table from the database as a pandas DataFrame.
            Args:
                table (str): The name of the table.
            Returns:
                pandas.DataFrame: The exported table as a DataFrame.

        run_query(self, q)
            Executes the provided SQL query and returns the result as a pandas DataFrame.
            Args:
                q (str): The SQL query to be executed.
            Returns:
                pandas.DataFrame: The result of the query as a DataFrame.
    """
    def __init__(self):
        self.query = None
        self.db = None
        self.conn = None
        self.cursor = None
        self.engine = None

    def connect(self):
        if self.engine is None:
            self.engine = sql.create_engine(f'sqlite:///{self.db}')
        self.conn = self.engine.connect()

    def save_data(self, name, data, replace=False):
        try:
            self.connect()
            if replace:
                data.to_sql(name, con=self.conn, if_exists='replace', index=False)
            else:
                data.to_sql(name, con=self.conn, if_exists='append', index=False)

            self.conn.close()

            return 0
        except Exception as e:
            print(data)
            return f"save_data to {name}: Error: {e}"

    def export_table(self, table):
        self.connect()
        df = pd.read_sql_table(table, self.conn)
        self.conn.close()

        return df

    def run_query(self, q):
        self.connect()

        df = pd.read_sql_query(q, self.conn)

        self.conn.close()

        return df
