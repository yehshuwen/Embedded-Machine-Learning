import serial
import time
import os

# Setting
serial_port = '/dev/cu.usbmodem13301'
baud_rate = 115200
data_dir = 'data'
collect_round = 3

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

def record_posture(posture_name, duration_time, round):
    file_name = f"{data_dir}/{posture_name}_{round}.csv"
    print(f"Ready to record: {posture_name} (lasting {duration_time}秒)")

    try:
        ser = serial.Serial(serial_port, baud_rate, timeout=1)
        print("Has connected to Arduino")

        ser.flushInput()

        input("Ready to record. Press ENTER to start...")

        ser.write(b's')
        ser.flush()

        print("Recording...")
        start_time = time.time()
        line_count = 0

        with open(file_name, 'w') as f:
            f.write("ax,ay,az,gx,gy,gz\n")

            while (time.time() - start_time) < duration_time:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8').strip()
                    if line:
                        f.write(line + "\n")
                        line_count += 1
        print(f"End recording. Save {line_count} rows to {file_name}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial port is closed")

if __name__ == "__main__":
    for i in range(collect_round):
        print(f"--- {i+1}/3 round ---")
        record_posture(posture_name="sitting", duration_time=10, round=i+1)
        record_posture(posture_name="supine", duration_time=10, round=i+1)
        record_posture(posture_name="prone", duration_time=10, round=i+1)
        record_posture(posture_name="side", duration_time=10, round=i+1)
        record_posture(posture_name="unknown", duration_time=10, round=i+1)

    print("\nEnd Collecting")