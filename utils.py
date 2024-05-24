import os
from datetime import datetime
import pytz
import pymongo
from pymongo import MongoClient


MONGO_DB_URL = "mongodb+srv://sriram:Ayynar%40123@msd.ywfrjgy.mongodb.net/?retryWrites=true&w=majority"


def get_collection(db_name, collection_name):
    client = MongoClient(MONGO_DB_URL)
    db = client[db_name]
    collection = db[collection_name]

    return collection


def store_in_mongodb(encoded_val, file_name):
    # Get the collection
    collection = get_collection(db_name = "bosch", collection_name = "images_v1")

    # Check if the document with the same file_name already exists
    existing_document = collection.find_one({"file_name": file_name})

    if existing_document:
        # Document with the same file_name already exists, delete it
        collection.delete_one({"file_name": file_name})
        print(f"Existing document with file_name {file_name} deleted.")

    # Define the document to be inserted
    document = {
        "encoded_val": encoded_val,
        "file_name": file_name
    }

    # Insert the document into the collection
    result = collection.insert_one(document)

    # Return the inserted document's ID
    return result.inserted_id


def get_file_details(file_name):
    # Get the collection
    collection = get_collection(db_name = "bosch", collection_name = "images_v1")

    # Query the collection for the document with the specified file_name
    document = collection.find_one({"file_name": file_name}, {"_id": 0, "file_name": 1, "encoded_val": 1})

    # Check if the document was found and return the relevant details
    if document:
        return document
    else:
        return None


def delete_all_data_from_collection(db_name = "bosch", collection_name = "images_v1"):
    # Get the collection
    collection = get_collection(db_name, collection_name)
    
    # Delete all documents in the collection
    result = collection.delete_many({})
    
    # Return the count of deleted documents
    return result.deleted_count


def create_collection(collection_name):
    # Connect to MongoDB Atlas
    client = pymongo.MongoClient(MONGO_DB_URL)

    # Access the specified database
    db = client["bosch"]

    # Create the new collection
    collection = db[collection_name]

    print(f"Collection '{collection_name}' created successfully.")

    # Close the connection
    client.close()


def insert_data(user_id, session_id, query, response, collection = get_collection(db_name = "bosch", collection_name = "chat_history_v1")):
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
            "timestamp": current_time_ist.strftime("%Y-%m-%d %H:%M:%S %Z%z")
        }
    }

    # Insert the document into the collection
    collection.insert_one(data_to_insert)


def get_latest_data(user_id, session_id, collection = get_collection(db_name = "bosch", collection_name = "chat_history_v1")):
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

        for document in cursor:
            # timestamp_ist = datetime.strptime(document["chat_history"]["timestamp"], "%Y-%m-%d %H:%M:%S %Z")
            # timestamp_ist = timestamp_ist.replace(tzinfo=ist_timezone)

            data_entry = {
                "query": document["chat_history"]["query"],
                "response": document["chat_history"]["response"],
                "timestamp": document["chat_history"]["timestamp"]
            }

            latest_data.append(data_entry)

        return latest_data
    
    except:
        return []


def get_full_data(user_id, session_id, collection = get_collection(db_name = "bosch", collection_name = "chat_history_v1")):
    try:
        ist_timezone = pytz.timezone('Asia/Kolkata')

        # Find the documents for the given user_id and session_id, sort by timestamp in descending order, and limit to 5
        cursor = collection.find(
            {"$and": [
                {"user_id": user_id},
                {"session_id": session_id}
            ]}
        ).sort("chat_history.timestamp", pymongo.DESCENDING)


        latest_data = []

        for document in cursor:
            # timestamp_ist = datetime.strptime(document["chat_history"]["timestamp"], "%Y-%m-%d %H:%M:%S %Z")
            # timestamp_ist = timestamp_ist.replace(tzinfo=ist_timezone)

            data_entry = {
                "query": document["chat_history"]["query"],
                "response": document["chat_history"]["response"]
            }

            latest_data.append(data_entry)

        return latest_data
    
    except:
        return []


def check_and_delete_existing_records(user_id, session_id, collection=None):
    if collection is None:
        collection = get_collection(db_name = "bosch", collection_name = "chat_history_v1")
    
    if collection is None:
        print("Collection is not available.")
        return False

    try:
        query = {"user_id": user_id, "session_id": session_id}
        count = collection.count_documents(query)
        
        if count > 0:
            collection.delete_many(query)
            print(f"Deleted {count} existing records for user_id: {user_id} and session_id: {session_id}.")
            return True
        else:
            print("No existing records found.")
            return False
    except Exception as e:
        print(f"Error checking and deleting records: {e}")
        return False