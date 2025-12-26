MAPS = {
    "L" : "A6 A3 A0 A7 A4 A1 A8 A5 A2 F8 - - F5 - - F2 - - B0 - - B3 - - B6 - - C0 - - C3 - - C6 - - - - - - - - - - - - - D6 - - D3 - - D0",
    "L'" : "A2 A5 A8 A1 A4 A7 A0 A3 A6 C0 - - C3 - - C6 - - D0 - - D3 - - D6 - - F8 - - F5 - - F2 - - - - - - - - - - - - - B6 - - B3 - - B0",
    "L2" : "A8 A7 A6 A5 A4 A3 A2 A1 A0 D0 - - D3 - - D6 - - F8 - - F5 - - F2 - - B0 - - B3 - - B6 - - - - - - - - - - - - - C6 - - C3 - - C0",
    "U" : "C0 C1 C2 - - - - - - B6 B3 B0 B7 B4 B1 B8 B5 B2 E0 E1 E2 - - - - - - - - - - - - - - - F0 F1 F2 - - - - - - A0 A1 A2 - - - - - -",
    "U'" : "F0 F1 F2 - - - - - - B2 B5 B8 B1 B4 B7 B0 B3 B6 A0 A1 A2 - - - - - - - - - - - - - - - C0 C1 C2 - - - - - - E0 E1 E2 - - - - - -",
    "U2" : "E0 E1 E2 - - - - - - B8 B7 B6 B5 B4 B3 B2 B1 B0 F0 F1 F2 - - - - - - - - - - - - - - - A0 A1 A2 - - - - - - C0 C1 C2 - - - - - -",
    "F" : "- - D0 - - D1 - - D2 - - - - - - A8 A5 A2 C6 C3 C0 C7 C4 C1 C8 C5 C2 E6 E3 E0 - - - - - - C6 - - C7 - - C8 - - - - - - - - - - -",
    "F'" : "- - B8 - - B7 - - B6 - - - - - - E0 E3 E6 C2 C5 C8 C1 C4 C7 C0 C3 C6 A2 A5 A8 - - - - - - D2 - - D1 - - D0 - - - - - - - - - - -",
    "F2" : "- - E6 - - E3 - - E0 - - - - - - D2 D1 D0 C8 C7 C6 C5 C4 C3 C2 C1 C0 B8 B7 B6 - - - - - - A8 - - A5 - - A2 - - - - - - - - - - -",
    "D" : "- - - - - - F6 F7 F8 - - - - - - - - - - - - - - - A6 A7 A8 D6 D3 D0 D7 D4 D1 D8 D5 D2 - - - - - - C6 C7 C8 - - - - - - E6 E7 E8",
    "D'" : "- - - - - - C6 C7 C8 - - - - - - - - - - - - - - - E6 E7 E8 D2 D5 D8 D1 D4 D7 D0 D3 D6 - - - - - - F6 F7 F8 - - - - - - A6 A7 A8",
    "D2" : "- - - - - - E6 E7 E8 - - - - - - - - - - - - - - - F6 F7 F8 D8 D7 D6 D5 D4 D3 D2 D1 D0 - - - - - - A6 A7 A8 - - - - - - C6 C7 C8",
    "R" : "- - - - - - - - - - - C2 - - C5 - - C8 - - D2 - - D5 - - D8 - - F6 - - F3 - - F0 E6 E3 E0 E7 E4 E1 E8 E5 E2 B8 - - B5 - - B2 - -",
    "R'" : "- - - - - - - - - - - F6 - - F3 - - F0 - - B2 - - B5 - - B8 - - C2 - - C5 - - C8 E2 E5 E8 E1 E4 E7 E0 E3 E6 D8 - - D5 - - D2 - -",
    "R2" : "- - - - - - - - - - - D2 - - D5 - - D8 - - F6 - - F3 - - F0 - - B2 - - B5 - - B8 E8 E7 E6 E5 E4 E3 E2 E1 E0 C8 - - C5 - - C2 - -",
    "B" : "B2 - - B1 - - B0 - - E2 E5 E8 - - - - - - - - - - - - - - - - - - - - - A0 A3 A6 - - D8 - - D7 - - D6 F6 F3 F0 F7 F4 F1 F8 F5 F2",
    "B'" : "D6 - - D7 - - D8 - - A6 A3 A0 - - - - - - - - - - - - - - - - - - - - - E8 E5 E2 - - B0 - - B1 - - B2 F2 F5 F8 F1 F4 F7 F0 F3 F6",
    "B2" : "E8 - - E5 - - E2 - - D8 D7 D6 - - - - - - - - - - - - - - - - - - - - - B2 B1 B0 - - A6 - - A3 - - A0 F8 F7 F6 F5 F4 F3 F2 F1 F0"
}

face_codes = ["A", "B", "C", "D", "E", "F"]

def get_index_from_code(c):
    # c = Map Code, like A6, E2, -, etc.
    if c == "-":
        return -1
    else:
        face_factor = face_codes.index(c[0])
        face_offset = 9*face_factor
        code_index = face_offset + int(c[1])
        return code_index

