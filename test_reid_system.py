# test_reid_system.py
"""
Test script for the Multi-Video Person Re-Identification System
Run this after starting the main server to verify functionality.
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_connection():
    """Test if server is running"""
    print("🔍 Testing server connection...")
    try:
        response = requests.get(f"{BASE_URL}/status")
        if response.status_code == 200:
            print("✅ Server is running!")
            return True
        else:
            print(f"❌ Server responded with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Is it running?")
        print("   Start server with: python -m uvicorn main:app --reload")
        return False

def test_identity_endpoint():
    """Test identity management endpoints"""
    print("\n🔍 Testing identity endpoints...")
    
    # Get identities
    response = requests.get(f"{BASE_URL}/identities")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Identity endpoint working!")
        print(f"   Total identities: {data['statistics']['total_identities']}")
        print(f"   Cross-video matches: {data['statistics']['cross_video_identities']}")
        return True
    else:
        print(f"❌ Identity endpoint failed: {response.status_code}")
        return False

def test_status_endpoint():
    """Test status endpoint"""
    print("\n🔍 Testing status endpoint...")
    
    response = requests.get(f"{BASE_URL}/status")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Status endpoint working!")
        print(f"   Active videos: {data['active_videos']}")
        print(f"   Identity stats: {data['identity_stats']}")
        return True
    else:
        print(f"❌ Status endpoint failed: {response.status_code}")
        return False

def test_reset_identities():
    """Test reset functionality"""
    print("\n🔍 Testing reset identities...")
    
    response = requests.post(f"{BASE_URL}/reset_identities")
    if response.status_code == 200:
        print("✅ Reset endpoint working!")
        return True
    else:
        print(f"❌ Reset endpoint failed: {response.status_code}")
        return False

def display_instructions():
    """Display usage instructions"""
    print("\n" + "="*60)
    print("📖 SYSTEM READY - Next Steps:")
    print("="*60)
    print()
    print("1️⃣  Open your browser and go to:")
    print(f"    👉 {BASE_URL}/")
    print()
    print("2️⃣  Upload test videos:")
    print("    - Click 'Choose Files'")
    print("    - Select multiple .mp4 video files")
    print("    - Click 'Upload & Start Processing'")
    print()
    print("3️⃣  Watch the magic happen:")
    print("    - Real-time person detection")
    print("    - Global ID assignment")
    print("    - Cross-video matching")
    print()
    print("4️⃣  Check the Identity Table:")
    print("    - See all unique people")
    print("    - Cross-video appearances highlighted")
    print("    - First/last seen timestamps")
    print()
    print("="*60)
    print("💡 Tips:")
    print("="*60)
    print("• Use videos with the same people to test cross-video matching")
    print("• People should wear distinctive clothing for best results")
    print("• Good lighting improves Re-ID accuracy")
    print("• Adjust similarity threshold in api/reid.py if needed")
    print()
    print("📁 Test videos are in: vedios/ folder")
    print("   Available: 1.mp4, 3.mp4, 4.mp4, v1.mp4, v2.mp4")
    print()

def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("🧪 Multi-Video Person Re-ID System - Test Suite")
    print("="*60)
    
    tests = [
        test_connection,
        test_identity_endpoint,
        test_status_endpoint,
        test_reset_identities
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
        time.sleep(0.5)
    
    print("\n" + "="*60)
    print("📊 Test Results")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All tests passed!")
        display_instructions()
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)





