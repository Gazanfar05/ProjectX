import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime
import os

class DiabetesXMLParser:
    def __init__(self, xml_file_path: str):
        self.xml_file_path = xml_file_path
        self.patient_id = None
        self.data = {
            'glucose': [],
            'medications': [],
            'lifestyle': []
        }
    
    def parse_xml(self):
        """Parse XML file - Kaggle diabetes dataset format"""
        print(f"📖 Parsing: {os.path.basename(self.xml_file_path)}")
        
        try:
            tree = ET.parse(self.xml_file_path)
            root = tree.getroot()
            
            self.patient_id = root.get('id')
            
            self._parse_glucose_levels(root)
            self._parse_finger_sticks(root)
            self._parse_basal(root)
            self._parse_bolus(root)
            self._parse_temp_basal(root)
            self._parse_meals(root)
            self._parse_exercise(root)
            self._parse_sleep(root)
            
            print(f"   ✓ Patient {self.patient_id}: {len(self.data['glucose'])} glucose, {len(self.data['medications'])} meds, {len(self.data['lifestyle'])} lifestyle")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            raise
    
    def _parse_timestamp(self, ts_str):
        """Parse timestamp - handles MM-DD-YYYY HH:MM:SS format"""
        try:
            return datetime.strptime(ts_str, "%m-%d-%Y %H:%M:%S")
        except:
            try:
                return pd.to_datetime(ts_str)
            except:
                return None
    
    def _parse_glucose_levels(self, root):
        """Parse continuous glucose monitor readings"""
        glucose_elem = root.find('glucose_level')
        if glucose_elem is not None:
            for event in glucose_elem.findall('event'):
                try:
                    ts = event.get('ts')
                    value = event.get('value')
                    
                    if ts and value:
                        timestamp = self._parse_timestamp(ts)
                        if timestamp:
                            glucose_val = float(value)
                            self.data['glucose'].append({
                                'timestamp': timestamp,
                                'glucose_level': glucose_val,
                                'source': 'cgm'
                            })
                except:
                    pass
    
    def _parse_finger_sticks(self, root):
        """Parse fingerstick glucose readings"""
        finger_elem = root.find('finger_stick')
        if finger_elem is not None:
            for event in finger_elem.findall('event'):
                try:
                    ts = event.get('ts')
                    value = event.get('value')
                    
                    if ts and value:
                        timestamp = self._parse_timestamp(ts)
                        if timestamp:
                            glucose_val = float(value)
                            self.data['glucose'].append({
                                'timestamp': timestamp,
                                'glucose_level': glucose_val,
                                'source': 'fingerstick'
                            })
                except:
                    pass
    
    def _parse_basal(self, root):
        """Parse basal insulin"""
        basal_elem = root.find('basal')
        if basal_elem is not None:
            for event in basal_elem.findall('event'):
                try:
                    ts = event.get('ts')
                    value = event.get('value')
                    
                    if ts and value:
                        timestamp = self._parse_timestamp(ts)
                        if timestamp:
                            dosage = float(value)
                            self.data['medications'].append({
                                'timestamp': timestamp,
                                'medication_type': 'insulin_basal',
                                'medication_name': 'Basal',
                                'dosage': dosage
                            })
                except:
                    pass
    
    def _parse_bolus(self, root):
        """Parse bolus insulin - uses ts_begin and dose"""
        bolus_elem = root.find('bolus')
        if bolus_elem is not None:
            for event in bolus_elem.findall('event'):
                try:
                    ts_begin = event.get('ts_begin')
                    dose = event.get('dose')
                    
                    if ts_begin and dose:
                        timestamp = self._parse_timestamp(ts_begin)
                        if timestamp:
                            dosage = float(dose)
                            carb_input = event.get('bwz_carb_input')
                            
                            self.data['medications'].append({
                                'timestamp': timestamp,
                                'medication_type': 'insulin_bolus',
                                'medication_name': 'Bolus',
                                'dosage': dosage
                            })
                            
                            # Also add carb info as lifestyle event if available
                            if carb_input:
                                try:
                                    carbs = float(carb_input)
                                    self.data['lifestyle'].append({
                                        'timestamp': timestamp,
                                        'event_type': 'meal',
                                        'carbs': carbs,
                                        'duration_minutes': None,
                                        'intensity': None
                                    })
                                except:
                                    pass
                except:
                    pass
    
    def _parse_temp_basal(self, root):
        """Parse temp basal insulin"""
        temp_basal_elem = root.find('temp_basal')
        if temp_basal_elem is not None:
            for event in temp_basal_elem.findall('event'):
                try:
                    ts_begin = event.get('ts_begin')
                    value = event.get('value')
                    
                    if ts_begin and value:
                        timestamp = self._parse_timestamp(ts_begin)
                        if timestamp:
                            dosage = float(value)
                            self.data['medications'].append({
                                'timestamp': timestamp,
                                'medication_type': 'insulin_temp_basal',
                                'medication_name': 'Temp Basal',
                                'dosage': dosage
                            })
                except:
                    pass
    
    def _parse_meals(self, root):
        """Parse meal data"""
        meal_elem = root.find('meal')
        if meal_elem is not None:
            for event in meal_elem.findall('event'):
                try:
                    ts = event.get('ts')
                    carbs = event.get('carbs')
                    
                    if ts and carbs:
                        timestamp = self._parse_timestamp(ts)
                        if timestamp:
                            carb_val = float(carbs)
                            self.data['lifestyle'].append({
                                'timestamp': timestamp,
                                'event_type': 'meal',
                                'carbs': carb_val,
                                'duration_minutes': None,
                                'intensity': None
                            })
                except:
                    pass
    
    def _parse_exercise(self, root):
        """Parse exercise data"""
        exercise_elem = root.find('exercise')
        if exercise_elem is not None:
            for event in exercise_elem.findall('event'):
                try:
                    ts = event.get('ts')
                    duration = event.get('duration')
                    intensity = event.get('intensity')
                    
                    if ts:
                        timestamp = self._parse_timestamp(ts)
                        if timestamp:
                            dur_min = int(float(duration)) if duration else 30
                            intens = str(intensity) if intensity else 'medium'
                            
                            self.data['lifestyle'].append({
                                'timestamp': timestamp,
                                'event_type': 'exercise',
                                'carbs': None,
                                'duration_minutes': dur_min,
                                'intensity': intens
                            })
                except:
                    pass
    
    def _parse_sleep(self, root):
        """Parse sleep data"""
        sleep_elem = root.find('sleep')
        if sleep_elem is not None:
            for event in sleep_elem.findall('event'):
                try:
                    tbegin = event.get('tbegin') or event.get('ts_begin')
                    tend = event.get('tend') or event.get('ts_end')
                    
                    if tbegin:
                        timestamp = self._parse_timestamp(tbegin)
                        if timestamp:
                            # Calculate duration if tend exists
                            dur_min = 480  # default 8 hours
                            if tend:
                                try:
                                    tend_time = self._parse_timestamp(tend)
                                    dur_min = int((tend_time - timestamp).total_seconds() / 60)
                                except:
                                    pass
                            
                            self.data['lifestyle'].append({
                                'timestamp': timestamp,
                                'event_type': 'sleep',
                                'carbs': None,
                                'duration_minutes': dur_min,
                                'intensity': None
                            })
                except:
                    pass
    
    def to_dataframe(self):
        """Convert to DataFrames"""
        df_glucose = pd.DataFrame(self.data['glucose']) if self.data['glucose'] else pd.DataFrame()
        df_meds = pd.DataFrame(self.data['medications']) if self.data['medications'] else pd.DataFrame()
        df_lifestyle = pd.DataFrame(self.data['lifestyle']) if self.data['lifestyle'] else pd.DataFrame()
        
        if not df_glucose.empty:
            df_glucose = df_glucose.sort_values('timestamp').reset_index(drop=True)
            df_glucose = df_glucose.dropna(subset=['timestamp', 'glucose_level'])
            df_glucose = df_glucose[df_glucose['glucose_level'] > 0]
            df_glucose['user_id'] = self.patient_id
        
        if not df_meds.empty:
            df_meds['user_id'] = self.patient_id
        
        if not df_lifestyle.empty:
            df_lifestyle['user_id'] = self.patient_id
        
        return df_glucose, df_meds, df_lifestyle

def parse_multiple_xml_files(directory: str):
    """Parse all XML files in directory"""
    all_glucose = []
    all_meds = []
    all_lifestyle = []
    
    xml_files = sorted([f for f in os.listdir(directory) if f.endswith('.xml')])
    
    print(f"\n📂 Found {len(xml_files)} XML files\n")
    
    for xml_file in xml_files:
        file_path = os.path.join(directory, xml_file)
        parser = DiabetesXMLParser(file_path)
        
        try:
            parser.parse_xml()
            df_glucose, df_meds, df_lifestyle = parser.to_dataframe()
            
            if not df_glucose.empty:
                all_glucose.append(df_glucose)
            if not df_meds.empty:
                all_meds.append(df_meds)
            if not df_lifestyle.empty:
                all_lifestyle.append(df_lifestyle)
                
        except Exception as e:
            print(f"   ⚠️  Skipped\n")
    
    print()
    combined_glucose = pd.concat(all_glucose, ignore_index=True) if all_glucose else pd.DataFrame()
    combined_meds = pd.concat(all_meds, ignore_index=True) if all_meds else pd.DataFrame()
    combined_lifestyle = pd.concat(all_lifestyle, ignore_index=True) if all_lifestyle else pd.DataFrame()
    
    print(f"✓ Total records: {len(combined_glucose)} glucose, {len(combined_meds)} meds, {len(combined_lifestyle)} lifestyle\n")
    
    return combined_glucose, combined_meds, combined_lifestyle
