# note_generation/data_saver.py
import json

def save_summarized_groups(encounter_id: str, conversation: str, summarized_groups: dict, reference_note: str = None, filename: str = "data/output/soap_dataset.json"):
    """Save summarized groups for fine-tuning."""
    data = {
        "encounter_id": encounter_id,
        "fixed_role_dialogue": conversation,
        "summarized_groups": summarized_groups,
        "reference_note": reference_note
    }
    with open(filename, "a") as f:
        json.dump(data, f)
        f.write("\n")