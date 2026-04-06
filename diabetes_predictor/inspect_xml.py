import xml.etree.ElementTree as ET
import sys

def inspect_xml(filepath):
    """Inspect XML structure"""
    print(f"\n📖 Inspecting: {filepath}\n")
    
    tree = ET.parse(filepath)
    root = tree.getroot()
    
    print(f"Root tag: {root.tag}")
    print(f"Root attributes: {root.attrib}\n")
    
    print("First 10 child elements:")
    for i, child in enumerate(list(root)[:10]):
        print(f"  {i+1}. {child.tag}: {child.attrib}")
        if child.text and child.text.strip():
            print(f"     Text: {child.text.strip()[:50]}")
    
    print(f"\nTotal elements: {len(list(root))}")
    
    # Count element types
    tags = {}
    for elem in root.iter():
        tag = elem.tag
        tags[tag] = tags.get(tag, 0) + 1
    
    print("\nElement type counts:")
    for tag, count in sorted(tags.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"  {tag}: {count}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_xml.py <xml_file>")
        sys.exit(1)
    
    inspect_xml(sys.argv[1])
