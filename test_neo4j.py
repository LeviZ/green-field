from neo4j import GraphDatabase
import os

uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(uri, auth=(username, password))

def print_greeting(message):
    with driver.session() as session:
        # Update from write_transaction to execute_write
        greeting = session.execute_write(lambda tx: tx.run("CREATE (a:Greeting) "
                                                           "SET a.message = $message "
                                                           "RETURN a.message + ', from node ' + id(a)", message=message).single().value())
        print(greeting)

if __name__ == "__main__":
    print_greeting("Hello, Neo4j!")
    driver.close()