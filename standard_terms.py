import json

with open('cleaned_text/sorted_sections.json') as f:
    data = json.load(f)

for filename, content in data.items():
    content["terms"] = "In Front of/Front: towards you.\n- Behind/Back: away from you.\n- Up: towards the ceiling.\n- Down: towards the floor.\n- Stud: the bump on a LEGO brick. Example: A 2x1 brick has two studs on it.\n- Vertically: going from front to behind.\n- Horizontally: going from left to right.\n- Upright: pointing up towards the ceiling.\n- That one/ppp: previously placed piece.\n- Plate: piece with studs.\n- Tile: smooth piece without studs (unless otherwise specified)\n- Symmetrically: a mirror image. Example: If you place a 2x1 brick with technic connector on the front wall\nat the right, connector to the front, and then place another such piece symmetrically on the back wall, at\nthe right, the technic connector of the second piece should point to the back, since it will be placed\nsymmetrically.\n- Centered-vertically: even amount of space in front of and behind the piece\n- Centered-horizontally: even amount of space left and right of the piece.\n- Row: studs lined up horizontally (left to right/side to side).\n- Column: studs lined up upright or vertically (top to bottom/back to front)."

    # Convert list to string for specific keys
    for key in ["introduction", "sorting", "instructions", "abbreviations", "misc", "ads"]:
        if isinstance(content[key], list) and content[key]:
            content[key] = content[key][0]
        if isinstance(content[key], list) and not content[key]:
            content[key] = ""

with open('cleaned_text/sorted_sections_modified.json', 'w') as f:
    json.dump(data, f, indent=4)

print("Terms have been successfully replaced in the JSON data.")