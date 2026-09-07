def get_plant_retriver(db):
    """
    Function to get the retriever for the plant database.
    Args:
        db: The vector store database."""
    plant_info_retriever = db.as_retriever(search_type ="similarity", 
                                           search_kwargs = {"k": 3}) 
    return plant_info_retriever