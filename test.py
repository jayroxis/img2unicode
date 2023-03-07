

for i in range(256):
    try:
        print(f"{i:3} {i:02X} {chr(i)}", end='\t')
        if (i+1) % 8 == 0:
            print()
    except RuntimeError:
        break