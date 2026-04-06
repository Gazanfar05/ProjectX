import xml.etree.ElementTree as ET

def debug_xml(filepath):
    """Debug XML structure"""
    print(f"\n📖 Debugging: {filepath}\n")
    
    tree = ET.parse(filepath)
    root = tree.getroot()
    
    # Check glucose_level
    glucose_elem = root.find('glucose_level')
    if glucose_elem is not None:
        print(f"✓ Found glucose_level element")
        events = glucose_elem.findall('event')
        print(f"  Events: {len(events)}")
        if events:
            first_event = events[0]
            print(f"  First event attributes: {first_event.attrib}")
            print(f"  First event text: {first_event.text}")
    else:
        print("✗ No glucose_level element")
    
    # Check finger_stick
    finger_elem = root.find('finger_stick')
    if finger_elem is not None:
        print(f"\n✓ Found finger_stick element")
        events = finger_elem.findall('event')
        print(f"  Events: {len(events)}")
        if events:
            first_event = events[0]
            print(f"  First event attributes: {first_event.attrib}")
            print(f"  First event text: {first_event.text}")
    else:
        print("✗ No finger_stick element")
    
    # Check bolus
    bolus_elem = root.find('bolus')
    if bolus_elem is not None:
        print(f"\n✓ Found bolus element")
        events = bolus_elem.findall('event')
        print(f"  Events: {len(events)}")
        if events:
            first_event = events[0]
            print(f"  First event attributes: {first_event.attrib}")
            print(f"  First event text: {first_event.text}")
    else:
        print("✗ No bolus element")
    
    # Print all unique element types and sample
    print(f"\n\nAll element types with samples:")
    elem_types = {}
    for elem in root.iter():
        tag = elem.tag
        if tag not in elem_types:
            elem_types[tag] = elem
    
    for tag, elem in sorted(elem_types.items()):
        print(f"\n{tag}:")
        print(f"  Attributes: {elem.attrib}")
        if elem.text and elem.text.strip():
            print(f"  Text: {elem.text.strip()[:100]}")
        children = list(elem)
        if children:
            print(f"  Children: {len(children)}")
            first_child = children[0]
            print(f"    First child: {first_child.tag} - {first_child.attrib}")

if __name__ == "__main__":
    debug_xml('data/raw/559-ws-training.xml')
