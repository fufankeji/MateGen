from fastapi import Body, HTTPException

from MateGen.mateGenClass import MateGenClass


def get_mate_gen(
        api_key: str = Body(..., description="API key required for operation"),
        enhanced_mode: bool = Body(False, description="Enable enhanced mode"),
        knowledge_base_chat: bool = Body(False, description="Enable knowledge base chat"),
        kaggle_competition_guidance: bool = Body(False, description="Enable Kaggle competition guidance"),
        competition_name: str = Body(None, description="Name of the Kaggle competition if guidance is enabled"),
        knowledge_base_name: str = Body(None, description="Name of the knowledge_base_chat is enabled"),
) -> MateGenClass:
    if kaggle_competition_guidance and not competition_name:
        raise HTTPException(status_code=400,
                            detail="Competition name is required when Kaggle competition guidance is enabled.")
    if knowledge_base_chat and not knowledge_base_name:
        raise HTTPException(status_code=400,
                            detail="knowledge_base_name is required when knowledge_base_chat is enabled.")

    return MateGenClass(api_key, enhanced_mode, knowledge_base_chat, kaggle_competition_guidance, competition_name, knowledge_base_name)
