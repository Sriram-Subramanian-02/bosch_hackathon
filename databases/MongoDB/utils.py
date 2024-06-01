from datetime import datetime
import pytz
import pymongo
from pymongo import MongoClient

from databases.MongoDB.constants import MONGO_DB_URL


def get_collection(db_name, collection_name):
    """
    Get the MongoDB collection.

    Args:
        db_name (str): The name of the database.
        collection_name (str): The name of the collection.

    Returns:
        pymongo.collection.Collection: The MongoDB collection.
    """

    client = MongoClient(MONGO_DB_URL)
    db = client[db_name]
    collection = db[collection_name]

    return collection


def store_in_mongodb(encoded_val, file_name):
    """
    Store an encoded image value and its file name in MongoDB.

    If a document with the same file name exists, it is deleted before insertion.

    Args:
        encoded_val (str): The base64 encoded value of the image.
        file_name (str): The name of the file.

    Returns:
        pymongo.results.InsertOneResult: The result of the insert operation.
    """

    # Get the collection
    collection = get_collection(db_name="bosch", collection_name="images_v1")

    # Check if the document with the same file_name already exists
    existing_document = collection.find_one({"file_name": file_name})

    if existing_document:
        # Document with the same file_name already exists, delete it
        collection.delete_one({"file_name": file_name})
        print(f"Existing document with file_name {file_name} deleted.")

    # Define the document to be inserted
    document = {"encoded_val": encoded_val, "file_name": file_name}

    # Insert the document into the collection
    result = collection.insert_one(document)

    # Return the inserted document's ID
    return result.inserted_id


def get_file_details(file_name):
    """
    Retrieve file details from MongoDB based on file name.

    Args:
        file_name (str): The name of the file.

    Returns:
        dict: The document containing the file details, or None if not found.
    """

    # Get the collection
    collection = get_collection(db_name="bosch", collection_name="images_v1")

    # Query the collection for the document with the specified file_name
    document = collection.find_one(
        {"file_name": file_name}, {"_id": 0, "file_name": 1, "encoded_val": 1}
    )

    # Check if the document was found and return the relevant details
    if document:
        return document
    else:
        return None


def delete_all_data_from_collection(db_name="bosch", collection_name="images_v1"):
    """
    Delete all documents from a specified collection.

    Args:
        db_name (str): The name of the database. Defaults to "bosch".
        collection_name (str): The name of the collection. Defaults to "images_v1".

    Returns:
        int: The count of deleted documents.
    """

    # Get the collection
    collection = get_collection(db_name, collection_name)

    # Delete all documents in the collection
    result = collection.delete_many({})

    # Return the count of deleted documents
    return result.deleted_count


def create_collection(collection_name):
    """
    Create a new collection in the MongoDB database.

    Args:
        collection_name (str): The name of the new collection.
    """

    # Connect to MongoDB Atlas
    client = pymongo.MongoClient(MONGO_DB_URL)

    # Access the specified database
    db = client["bosch"]

    # Create the new collection
    collection = db[collection_name]

    print(f"Collection '{collection_name}' created successfully.")

    # Close the connection
    client.close()


def insert_data(user_id, session_id, query, response, is_probing_question, collection=get_collection(db_name="bosch", collection_name="chat_history_v1")):
    """
    Insert chat data into the MongoDB collection.
    Args:
        user_id (str): The user ID.
        session_id (str): The session ID.
        query (str): The query text.
        response (str): The response text.
        is_probing_question (bool): Tells us whether the query is a probing question or not
        collection (pymongo.collection.Collection): The MongoDB collection to insert data into. Defaults to "chat_history_v1".
    """
    # Get the current UTC time
    current_time_utc = datetime.utcnow()

    # Define the IST timezone
    ist_timezone = pytz.timezone('Asia/Kolkata')

    # Convert the UTC time to IST
    current_time_ist = current_time_utc.astimezone(ist_timezone)

    # Create a document to insert
    data_to_insert = {
        # "_id": ObjectId(),  # Use ObjectId to generate a unique _id for each document
        "user_id": user_id,
        "session_id": session_id,
        "chat_history": {
            "query": query,
            "response": response,
            "timestamp": current_time_ist.strftime("%Y-%m-%d %H:%M:%S %Z%z"),
            "is_probing_question": is_probing_question
        }
    }

    # Insert the document into the collection
    collection.insert_one(data_to_insert)

def get_latest_data(user_id, session_id, collection=get_collection(db_name="bosch", collection_name="chat_history_v1")):
    """
    Retrieve the latest chat data for a specific user and session, along with the latest probing question.

    Args:
        user_id (str): The user ID.
        session_id (str): The session ID.
        collection (pymongo.collection.Collection, optional): The MongoDB collection to retrieve data from. 
            Defaults to the "chat_history_v1" collection in the "bosch" database.

    Returns:
        tuple: A tuple containing:
            - latest_data (list): A list of dictionaries representing the latest chat data documents. Each dictionary contains:
                - query (str): The user's query.
                - response (str): The assistant's response.
                - is_probing_question (bool): A flag indicating if the query was a probing question.
            - probing_data (list): A list of dictionaries representing the latest probing question data. Each dictionary contains:
                - query (str): The user's query.
                - response (str): The assistant's response.
                - is_probing_question (bool): A flag indicating if the query was a probing question.
    """
    try:
        ist_timezone = pytz.timezone('Asia/Kolkata')

        # Find the documents for the given user_id and session_id, sort by timestamp in descending order, and limit to 5
        cursor = collection.find(
            {"$and": [
                {"user_id": user_id},
                {"session_id": session_id}
            ]}
        ).sort("chat_history.timestamp", pymongo.DESCENDING).limit(5)

        latest_data = []
        probing_data = []

        for document in cursor:
            # timestamp_ist = datetime.strptime(document["chat_history"]["timestamp"], "%Y-%m-%d %H:%M:%S %Z")
            # timestamp_ist = timestamp_ist.replace(tzinfo=ist_timezone)

            data_entry = {
                "query": document["chat_history"]["query"],
                "response": document["chat_history"]["response"],
                "is_probing_question": document["chat_history"]["is_probing_question"]
            }

            latest_data.append(data_entry)
        
        flag = True

        for doc in latest_data:
            if doc['is_probing_question'] and flag:
                probing_data.append(doc)
            else:
                flag=False
        
        return latest_data, probing_data
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return [], []


def get_full_data(
    user_id,
    session_id,
    collection=get_collection(db_name="bosch", collection_name="chat_history_v1"),
):
    """
    Retrieve the full chat data for a specific user and session.

    Args:
        user_id (str): The user ID.
        session_id (str): The session ID.
        collection (pymongo.collection.Collection): The MongoDB collection to retrieve data from. Defaults to "chat_history_v1".

    Returns:
        list: A list of all chat data documents.
    """

    try:
        ist_timezone = pytz.timezone("Asia/Kolkata")

        # Find the documents for the given user_id and session_id, sort by timestamp in descending order, and limit to 5
        cursor = collection.find(
            {"$and": [{"user_id": user_id}, {"session_id": session_id}]}
        ).sort("chat_history.timestamp", pymongo.DESCENDING)

        latest_data = []

        for document in cursor:
            # timestamp_ist = datetime.strptime(document["chat_history"]["timestamp"], "%Y-%m-%d %H:%M:%S %Z")
            # timestamp_ist = timestamp_ist.replace(tzinfo=ist_timezone)

            data_entry = {
                "query": document["chat_history"]["query"],
                "response": document["chat_history"]["response"],
            }

            latest_data.append(data_entry)

        return latest_data

    except:
        return []


def check_and_delete_existing_records(user_id, session_id, collection=None):
    """
    Check for existing records in the MongoDB collection and delete them if found.

    Args:
        user_id (str): The user ID.
        session_id (str): The session ID.
        collection (pymongo.collection.Collection): The MongoDB collection to check. Defaults to "chat_history_v1".

    Returns:
        bool: True if records were found and deleted, False otherwise.
    """

    if collection is None:
        collection = get_collection(db_name="bosch", collection_name="chat_history_v1")

    if collection is None:
        print("Collection is not available.")
        return False

    try:
        query = {"user_id": user_id, "session_id": session_id}
        count = collection.count_documents(query)

        if count > 0:
            collection.delete_many(query)
            print(
                f"Deleted {count} existing records for user_id: {user_id} and session_id: {session_id}."
            )
            return True
        else:
            print("No existing records found.")
            return False
    except Exception as e:
        print(f"Error checking and deleting records: {e}")
        return False
