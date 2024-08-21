from fastapi import Body, HTTPException

from MateGen.mateGenClass import MateGenClass
from fastapi import HTTPException
import os
from dotenv import load_dotenv
from openai import OpenAI
from MateGen.mateGenClass import decrypt_string


def get_mate_gen(
        api_key: str = Body(..., description="API key required for operation"),
        enhanced_mode: bool = Body(False, description="Enable enhanced mode"),
        knowledge_base_chat: bool = Body(False, description="Enable knowledge base chat"),
        kaggle_competition_guidance: bool = Body(False, description="Enable Kaggle competition guidance"),
        competition_name: str = Body(None, description="Name of the Kaggle competition if guidance is enabled"),
        knowledge_base_name: str = Body(None, description="Name of the knowledge_base_chat is enabled"),
) -> MateGenClass:
    """
    用于生成 MateGen Class 的实例
    :param api_key:
    :param enhanced_mode:
    :param knowledge_base_chat:
    :param kaggle_competition_guidance:
    :param competition_name:
    :param knowledge_base_name:
    :return:
    """
    if kaggle_competition_guidance and not competition_name:
        raise HTTPException(status_code=400,
                            detail="Competition name is required when Kaggle competition guidance is enabled.")
    if knowledge_base_chat and not knowledge_base_name:
        raise HTTPException(status_code=400,
                            detail="knowledge_base_name is required when knowledge_base_chat is enabled.")

    return MateGenClass(api_key, enhanced_mode, knowledge_base_chat, kaggle_competition_guidance, competition_name,
                        knowledge_base_name)


def get_openai_instance(
        api_key: str = Body(..., description="API key required for operation")
) -> OpenAI:
    """
    用于生成原始的 OpenAI  实例
    :param api_key:
    :return:
    """
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        dotenv_path = os.path.join(BASE_DIR, '.env')

        load_dotenv(dotenv_path)
        base_url = os.getenv('BASE_URL')
        if not base_url:
            raise ValueError("BASE_URL not found in the environment variables.")

        original_string = decrypt_string(api_key, key=b'YAboQcXx376HSUKqzkTz8LK1GKs19Skg4JoZH4QUCJc=')
        if not original_string:
            raise ValueError("Failed to decrypt API key.")

        split_strings = original_string.split(' ')
        if not split_strings or len(split_strings) == 0:
            raise ValueError("Decrypted API key is invalid or empty.")

        s1 = split_strings[0]
        if not s1:
            raise ValueError("API key is empty after decryption and splitting.")

        return OpenAI(api_key=s1, base_url=base_url)

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")



