
# Open stream with the index of the chosen device you selected from your initial code
stream = p.open(format=p.get_format_from_width(width=2),
                channels=1,
                output=True,
                rate=OUTPUT_SAMPLE_RATE,
                input_device_index=INDEX_OF_CHOSEN_INPUT_DEVICE, # This is where you specify which input device to use
                stream_callback=callback)

# Start processing and do whatever else...
stream.start_stream()
