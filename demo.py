import os
import sys

def main():
    print("🔧 SURFACE DEFECT DETECTION SYSTEM DEMO")
    print("=" * 50)
    print("🎯 6 Defect Classes Supported")
    print("=" * 50)
    
    # Check if test images exist
    test_dir = "test_images"
    if not os.path.exists(test_dir):
        print("❌ Test images directory not found!")
        return
    
    # Run detection on sample images
    print("🚀 Running detection on sample images...")
    print()
    
    # Import and run main detection
    import subprocess
    result = subprocess.run([sys.executable, "main.py", test_dir], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print(result.stdout)
        print("✅ Demo completed successfully!")
        print("\n💡 To run on your own images:")
        print("   python main.py path/to/your/images/")
        print("   python main.py single_image.jpg")
    else:
        print("❌ Demo failed:")
        print(result.stderr)

if __name__ == "__main__":
    main()
