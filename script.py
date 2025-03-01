import sys

def main():
    user_input = sys.argv[1]  # Read input from Node.js
    response = f"Python received: {user_input}"
    print(response)  # Send output back to Node.js

if __name__ == "__main__":
    main()
