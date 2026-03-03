"""
Batch transcription client — sends HoP_{episode}_*.mp3 files to a transcription server.

Usage:
    python batch_transcribe_client.py --model_name whisper --episode 001
    python batch_transcribe_client.py --model_name granite --episode 002
    python batch_transcribe_client.py --model_name canary --episode 005
"""
import os
import re
import argparse
import requests


def get_sort_key(filename, episode):
    """Extracts the trailing number from 'HoP_{episode}_xxx.mp3' for correct sorting."""
    # Use an f-string to inject the specific episode number into the regex pattern
    match = re.search(fr'HoP_{episode}_(\d+)', filename)
    if match:
        return int(match.group(1))
    return float('inf')


def main():
    parser = argparse.ArgumentParser(
        description="Batch transcription client for HoP audio files"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model identifier used in the output filename and port routing (e.g., whisper, granite, canary)",
    )
    parser.add_argument(
        "--episode",
        type=str,
        required=True,
        help="The 3-digit episode number to process (e.g., 001, 002)",
    )
    parser.add_argument(
        "--server_url",
        type=str,
        default=None,
        help="Optional: Full URL of the /transcribe endpoint. If omitted, it is inferred from the model_name.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./data/",
        help="Directory containing MP3 files (default: ./data/)",
    )
    args = parser.parse_args()

    # Determine port based on model name
    model_ports = {
        "whisper": 8000,
        "granite": 8001,
        "canary": 8002
    }

    # Set the server URL automatically if not manually provided
    if args.server_url:
        server_url = args.server_url
    else:
        # Default to 8000 if an unknown model name is passed
        port = model_ports.get(args.model_name.lower(), 8000)
        server_url = f"http://localhost:{port}/transcribe"

    # Build output directory and create it if it doesn't exist
    output_dir = os.path.join("output", args.model_name.lower(), args.episode)
    os.makedirs(output_dir, exist_ok=True)

    # Build final output filepath
    filename = f"transcription_results_ep{args.episode}_{args.model_name.lower()}.txt"
    output_filepath = os.path.join(output_dir, filename)

    # 1. Get and sort files dynamically based on the episode passed
    try:
        all_files = os.listdir(args.input_dir)
        hop_files = [
            f for f in all_files
            if f.startswith(f"HoP_{args.episode}_") and f.endswith(".mp3")
        ]
        
        # Pass the episode into the sort key using a lambda function
        sorted_files = sorted(hop_files, key=lambda f: get_sort_key(f, args.episode))

        if not sorted_files:
            print(f"No matching HoP_{args.episode}_*.mp3 files found in {args.input_dir}.")
            return

        print(f"Found {len(sorted_files)} files for Episode {args.episode}. Starting transcription...")
        print(f"  Model:  {args.model_name}")
        print(f"  Server: {server_url}")
        print(f"  Output: {output_filepath}\n")

        # 2. Process files and save results
        with open(output_filepath, "w", encoding="utf-8") as out_f:
            for filename in sorted_files:
                file_path = os.path.join(args.input_dir, filename)
                print(f"Processing: {filename}...", end=" ", flush=True)

                try:
                    with open(file_path, "rb") as f:
                        files = {"file": f}
                        response = requests.post(server_url, files=files)

                        if response.status_code == 200:
                            result_json = response.json()
                            transcribed_text = result_json.get("text", "")
                            out_f.write(f"[{filename}]\n{transcribed_text}\n\n")
                            print("Done.")
                        else:
                            error_msg = f"Error {response.status_code}: {response.text}"
                            print("Failed.")
                            out_f.write(f"[{filename}]\nFAILED: {error_msg}\n\n")

                except requests.exceptions.ConnectionError:
                    print(f"\nConnection Error: Is the server running at {server_url}?")
                    break
                except Exception as e:
                    print(f"Error: {e}")

        print(f"\n--- Processing Complete ---")
        print(f"Results saved to: {os.path.abspath(output_filepath)}")

    except FileNotFoundError:
        print(f"Error: Directory '{args.input_dir}' not found.")


if __name__ == "__main__":
    main()