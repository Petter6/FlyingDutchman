# stats.py - Global statistics tracking (Singleton)
from collections import Counter
import numpy as np
import os 


class GlobalStats:
    _instance = None  # Singleton instance

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalStats, cls).__new__(cls)
            cls._instance.reset()  # Initialize stats
        return cls._instance

    def reset(self):
        """Resets all tracked statistics"""
        self.stats = {
            "occ_px": 0,
            "tot_px": 0,
            "occ_oob": 0,
            "occ_obj": 0,
            "occ_self": 0,
            "disp_hist": Counter(),
            "render_time": 0.0,
            "exr_time": 0.0,
            "disp_time": 0.0,
            "create_objects_time": 0.0,
            "color_diff_r": 0,
            "color_diff_g": 0,
            "color_diff_b": 0
        }

    def update(self, key, value):
        """Updates a specific stat"""
        if key in self.stats:
            if isinstance(self.stats[key], (int, float)):
                self.stats[key] += value
            elif isinstance(self.stats[key], set):
                self.stats[key].add(value)
            elif isinstance(self.stats[key], Counter):
                self.stats[key].update(value)
        else:
            print(f"⚠️ Warning: '{key}' is not a recognized stat.")

    def report(self, config=None):
        print("\n📊 Statistieken:")
        print(f"🔹 Totale pixels:     {self.stats['tot_px']}")
        print(f"🔹 Occluded pixels:   {self.stats['occ_px']}")
        print(f"   └─ Buiten beeld:   {self.stats['occ_oob']}")
        print(f"   └─ Door objecten:  {self.stats['occ_obj']}")
        print(f"   └─ Eigen rotatie:  {self.stats['occ_self']}")
        print(f"⏱️  Render tijd:      {self.stats['render_time']:.2f} s")
        print(f"⏱️  EXR verwerking:   {self.stats['exr_time']:.2f} s")
        print(f"⏱️  Displacement tijd:{self.stats['disp_time']:.2f} s")
        print(f"⏱️  Object creatie:   {self.stats['create_objects_time']:.2f} s")

        # Kleurverschillen (gemiddelde per pixel)
        tot_px = max(self.stats['tot_px'], 1)  # Bescherming tegen deling door nul
        avg_r = self.stats["color_diff_r"] / tot_px
        avg_g = self.stats["color_diff_g"] / tot_px
        avg_b = self.stats["color_diff_b"] / tot_px

        print(f"\n🎨 Gemiddelde kleurverschillen per kanaal:")
        print(f"   🔴 R: {avg_r:.2f}")
        print(f"   🟢 G: {avg_g:.2f}")
        print(f"   🔵 B: {avg_b:.2f}")

            
            

# Create a global stats instance
global_stats = GlobalStats()