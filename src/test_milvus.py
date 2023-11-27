from milvus import VDBHandeler
from pymilvus import CollectionSchema, FieldSchema, DataType, connections, db, utility, Collection, Index
from types import SimpleNamespace
import numpy as np
import uuid
import logging
import time
import llm


def create_table(vdb: VDBHandeler):
    memory_id = FieldSchema(
        name="id",
        dtype=DataType.VARCHAR,
        max_length=200,
        is_primary=True,
    )
    agent_id = FieldSchema(
        name="playerId",
        dtype=DataType.VARCHAR,
        max_length=200,
        # The default value will be used if this field is left empty during data inserts or upserts.
    )
    embeddings = FieldSchema(
        name="values",
        dtype=DataType.FLOAT_VECTOR,
        dim=1536
    )
    private_text = FieldSchema(
        name="private_text",
        dtype=DataType.VARCHAR,
        max_length=2048,
        description="private_text"
    )

    schema = CollectionSchema(
        fields=[memory_id, embeddings, agent_id, private_text],
        description="test_table",
        enable_dynamic_field=True
    )
    data = {}
    data["database_name"] = "test_db"
    data["table_name"] = "test_table"
    data["schema"] = schema
    f_n = "values"
    data["reset"] = True
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 256}
    }
    fake_index = SimpleNamespace()
    setattr(fake_index, "field_name", f_n)
    setattr(fake_index, "params", index_params)
    data["indexes"] = [fake_index]
    vdb.create_table(data)


def test_insert_data(vdb: VDBHandeler):
    data = {}
    data["database_name"] = "test_db"
    data["table_name"] = "test_table"
    data_id = str(uuid.uuid4())
    upsert_data = [[data_id], [np.random.normal(0, 0.05, 1536)], [
        "agent1"], ["QAQ!!"]]
    data["upsert_data"] = upsert_data
    vdb.upsert_Data(data)


def test_query_data(vdb: VDBHandeler):
    data = {}
    data["database_name"] = "test_db"
    data["table_name"] = "test_table"

    data["search_params"] = {
        "metric_type": "L2",
        "offset": 1,
        "ignore_growing": False,
        "params": {"nprobe": 4}
    }
    data["topK"] = 5
    data["embedding"] = np.random.normal(0, 0.03, 1536)
    data["search_field"] = "values"
    r = vdb.query_Data(data)
    rr = r["result"]
    # logging.info(rr)
    logging.info(rr[0])
    logging.info(rr[0][0].entity)
    logging.info(rr[0][0].to_dict()["entity"]["private_text"])


def test_delete_data(vdb: VDBHandeler, delete_all=False, delete_id=None):
    data = {}
    data["database_name"] = "test_db"
    data["table_name"] = "test_table"
    if delete_all:
        data["delete_all"] = True
        vdb.delete_Data(data)
        return
    else:
        data["delete_id"] = delete_id
        vdb.delete_Data(data)


def test_openai_insert(vdb: VDBHandeler, text: str):
    embeds = llm.get_embeddings(text)["embeddings"]
    # logging.info(len(embeds))
    data = {}
    data["database_name"] = "test_db"
    data["table_name"] = "test_table"
    data_id = str(uuid.uuid4())
    upsert_data = [[data_id], [embeds], [
        "agent1"], [text]]
    data["upsert_data"] = upsert_data
    vdb.upsert_Data(data)


def test_openai_search(vdb: VDBHandeler, text: str):
    embeds = llm.get_embeddings(text)["embeddings"]
    logging.info(len(embeds))
    data = {}
    data["database_name"] = "test_db"
    data["table_name"] = "test_table"

    data["search_params"] = {
        "metric_type": "L2",
        "params": {"nprobe": 4}
    }
    data["topK"] = 7
    data["embedding"] = embeds
    data["search_field"] = "values"
    r = vdb.query_Data(data)["result"]
    logging.info(r)
    res_text = [x.to_dict()["entity"]["private_text"] for x in r[0]]
    score = [x.to_dict()["distance"] for x in r[0]]

    logging.info(list(zip(score, res_text)))


def reset_database(vdb):
    test_delete_data(vdb, delete_all=True)
    create_table(vdb)
    vdb.print_records("test_db")
    # time.sleep(2)


def insert_records():
    test_openai_insert(vdb, "Every morning, I eat toast for breakfast.")

    test_openai_insert(vdb, "I usually have toast for Lunch.")

    test_openai_insert(vdb, "I prefer reading books in the evening.")
    test_openai_insert(
        vdb, "In the evening, I enjoy curling up with a good book.")

    test_openai_insert(vdb, "Soccer is my favorite sport to play.")
    test_openai_insert(vdb, "I love playing soccer with my friends.")
    pass


if __name__ == '__main__':
    vdb = VDBHandeler()
    # vdb.flush()
    reset_database(vdb)

    # test_insert_data(vdb)
    # test_query_data(vdb)
    insert_records()

    # vdb.print_records("test_db")

    test_openai_search(vdb, "Toast is my go-to breakfast choice.")
