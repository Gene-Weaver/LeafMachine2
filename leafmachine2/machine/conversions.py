
import numpy as np

def test_conversion(current_unit, current_val, challenge_unit, challenge_val):
    current_cm = convert_to_cm(current_unit, current_val)
    challenge_cm = convert_to_cm(challenge_unit, challenge_val)

    if current_cm == challenge_cm:
        return True
    else:
        return False


def convert_to_cm(unit, unit_value):
        unit_value_converted = []

        if unit == '32nd':
            for val in unit_value:
                unit_value_converted.append(float(np.multiply(np.multiply(val, 32), 2.54)))
        elif unit == '16th':
            for val in unit_value:
                unit_value_converted.append(float(np.multiply(np.multiply(val, 16), 2.54)))
        elif unit == '8th':
            for val in unit_value:
                unit_value_converted.append(float(np.multiply(np.multiply(val, 8), 2.54)))
        elif unit == '4th':
            for val in unit_value:
                unit_value_converted.append(float(np.multiply(np.multiply(val, 4), 2.54)))
        elif unit == 'halfinch':
            for val in unit_value:
                unit_value_converted.append(float(np.multiply(np.multiply(val, 2), 2.54)))
        elif unit == 'inch':
            for val in unit_value:
                unit_value_converted.append(float(np.multiply(np.multiply(val, 1), 2.54)))

        elif unit == 'halfmm':
            for val in unit_value:
                unit_value_converted.append(float(np.multiply(val, 20)))
        elif unit == 'mm':
            for val in unit_value:
                unit_value_converted.append(float(np.multiply(val, 10)))
        elif unit == '4thcm':
            for val in unit_value:    
                unit_value_converted.append(float(np.multiply(val, 4)))
        elif unit == 'halfcm':
            for val in unit_value:
                unit_value_converted.append(float(np.multiply(val, 2)))
        elif unit == 'cm':
            for val in unit_value:
                unit_value_converted.append(float(np.multiply(val, 1)))

        return unit_value_converted


def is_within_tolerance_mm(self, candidate, unknown, ):
        tol = 0.01  # 2.5% tolerance
        upper_bound = candidate * (1 + tol)
        lower_bound = candidate * (1 - tol)

        if self.is_dual:
            if lower_bound <= float(np.multiply(np.multiply(unknown, 32), 25.4)) <= upper_bound:     #from_32nd_to_inch_to_cm
                return 'mm', '32nd'
            elif lower_bound <= float(np.multiply(np.multiply(unknown, 16), 25.4)) <= upper_bound:         #from_16th_to_inch_to_cm
                return 'mm', '16th'
            elif lower_bound <= float(np.multiply(np.multiply(unknown, 8), 25.4)) <= upper_bound:       #from_8th_to_inch_to_cm
                return 'mm', '8th'
            elif lower_bound <= float(np.multiply(np.multiply(unknown, 4), 25.4)) <= upper_bound:       #from_4th_to_inch_to_cm
                return 'mm', '4th'
            elif lower_bound <= float(np.multiply(np.multiply(unknown, 2), 25.4)) <= upper_bound:      #from_halfinch_to_inch_to_cm
                return 'mm', 'halfinch'
            elif lower_bound <= float(np.multiply(np.multiply(unknown, 1), 25.4)) <= upper_bound:          #from_inch_to_inch_to_cm
                return 'mm', 'inch'
            
            # TODO buildout the other options
            elif lower_bound <= float(np.multiply(unknown, 32)) <= upper_bound:         #from_32nd_to_inch
                return 'inch', '32nd'
            elif lower_bound <= float(np.multiply(unknown, 16)) <= upper_bound:     #from_16th_to_inch
                return 'inch', '16th'
            elif lower_bound <= float(np.multiply(unknown, 8)) <= upper_bound:       #from_8th_to_inch
                return 'inch', '8th'
            elif lower_bound <= float(np.multiply(unknown, 4)) <= upper_bound:       #from_4th_to_inch
                return 'inch', '4th'
            elif lower_bound <= float(np.multiply(unknown, 2)) <= upper_bound:      #from_halfinch_to_inch
                return 'inch', 'halfinch'
            # elif lower_bound <= float(np.multiply(unknown, 1)) <= upper_bound:      #from_inch_to_inch
            #     return 'inch', 'inch'

            elif lower_bound <= float(np.multiply(unknown, 2)) <= upper_bound:          #from_halfmm_to_cm
                return 'mm', 'halfmm'
            # elif lower_bound <= float(np.multiply(unknown, 1)) <= upper_bound:         #from_mm_to_cm
            #     return 'mm', 'mm'
            elif lower_bound <= float(np.multiply(unknown, 2.5)) <= upper_bound:         #from_4th_to_cm
                return 'mm', '4thcm'
            elif lower_bound <= float(np.multiply(unknown, 5)) <= upper_bound:       #from_halfcm_to_cm
                return 'mm', 'halfcm'
            elif lower_bound <= float(np.multiply(unknown, 10)) <= upper_bound:      #from_cm_to_cm
                return 'mm', 'cm'

            elif lower_bound <= float(np.multiply(unknown, 1)) <= upper_bound:


            else:
                return None, None

        # metric
        elif self.is_metric:
            if lower_bound <= float(np.multiply(unknown, 2)) <= upper_bound:          #from_halfmm_to_cm
                return 'mm', 'halfmm'
            elif lower_bound <= float(np.multiply(unknown, 1)) <= upper_bound:         #from_mm_to_cm
                return 'mm', 'mm'
            elif lower_bound <= float(np.multiply(unknown, 2.5)) <= upper_bound:         #from_4th_to_cm
                return 'mm', '4thcm'
            elif lower_bound <= float(np.multiply(unknown, 5)) <= upper_bound:       #from_halfcm_to_cm
                return 'mm', 'halfcm'
            elif lower_bound <= float(np.multiply(unknown, 10)) <= upper_bound:      #from_cm_to_cm
                return 'mm', 'cm'
            else:
                return None, None
            
        # TODO buildout the other options
        elif self.is_standard:
            if lower_bound <= float(np.multiply(unknown, 32)) <= upper_bound:         #from_32nd_to_inch
                return 'inch', '32nd'
            elif lower_bound <= float(np.multiply(unknown, 16)) <= upper_bound:     #from_16th_to_inch
                return 'inch', '16th'
            elif lower_bound <= float(np.multiply(unknown, 8)) <= upper_bound:       #from_8th_to_inch
                return 'inch', '8th'
            elif lower_bound <= float(np.multiply(unknown, 4)) <= upper_bound:       #from_4th_to_inch
                return 'inch', '4th'
            elif lower_bound <= float(np.multiply(unknown, 2)) <= upper_bound:      #from_halfinch_to_inch
                return 'inch', 'halfinch'
            elif lower_bound <= float(np.multiply(unknown, 1)) <= upper_bound:      #from_inch_to_inch
                return 'inch', 'inch'
            else:
                return None, None

    def is_within_tolerance_cm(self, candidate, unknown):
        tol = 0.01  # 2.5% tolerance
        upper_bound = candidate * (1 + tol)
        lower_bound = candidate * (1 - tol)

        if self.is_dual:
            if lower_bound <= float(np.multiply(np.multiply(unknown, 32), 2.54)) <= upper_bound:     #from_32nd_to_inch_to_cm
                return 'cm', '32nd'
            elif lower_bound <= float(np.multiply(np.multiply(unknown, 16), 2.54)) <= upper_bound:         #from_16th_to_inch_to_cm
                return 'cm', '16th'
            elif lower_bound <= float(np.multiply(np.multiply(unknown, 8), 2.54)) <= upper_bound:       #from_8th_to_inch_to_cm
                return 'cm', '8th'
            elif lower_bound <= float(np.multiply(np.multiply(unknown, 4), 2.54)) <= upper_bound:       #from_4th_to_inch_to_cm
                return 'cm', '4th'
            elif lower_bound <= float(np.multiply(np.multiply(unknown, 2), 2.54)) <= upper_bound:      #from_halfinch_to_inch_to_cm
                return 'cm', 'halfinch'
            elif lower_bound <= float(np.multiply(np.multiply(unknown, 1), 2.54)) <= upper_bound:          #from_inch_to_inch_to_cm
                return 'cm', 'inch'
            
            elif lower_bound <= float(np.multiply(unknown, 32)) <= upper_bound:         #from_32nd_to_inch
                return 'inch', '32nd'
            elif lower_bound <= float(np.multiply(unknown, 16)) <= upper_bound:     #from_16th_to_inch
                return 'inch', '16th'
            elif lower_bound <= float(np.multiply(unknown, 8)) <= upper_bound:       #from_8th_to_inch
                return 'inch', '8th'
            elif lower_bound <= float(np.multiply(unknown, 4)) <= upper_bound:       #from_4th_to_inch
                return 'inch', '4th'
            elif lower_bound <= float(np.multiply(unknown, 2)) <= upper_bound:      #from_halfinch_to_inch
                return 'inch', 'halfinch'
            elif lower_bound <= float(np.multiply(unknown, 1)) <= upper_bound:      #from_inch_to_inch
                return 'inch', 'inch'

            elif lower_bound <= float(np.multiply(unknown, 20)) <= upper_bound:          #from_halfmm_to_cm
                return 'cm', 'halfmm'
            elif lower_bound <= float(np.multiply(unknown, 10)) <= upper_bound:         #from_mm_to_cm
                return 'cm', 'mm'
            elif lower_bound <= float(np.multiply(unknown, 4)) <= upper_bound:         #from_4th_to_cm
                return 'cm', '4thcm'
            elif lower_bound <= float(np.multiply(unknown, 2)) <= upper_bound:       #from_halfcm_to_cm
                return 'cm', 'halfcm'
            elif lower_bound <= float(np.multiply(unknown, 1)) <= upper_bound:      #from_cm_to_cm
                return 'cm', 'cm'
            else:
                return None, None

        # metric
        elif self.is_metric:
            if lower_bound <= float(np.multiply(unknown, 20)) <= upper_bound:          #from_halfmm_to_cm
                return 'cm', 'halfmm'
            elif lower_bound <= float(np.multiply(unknown, 10)) <= upper_bound:         #from_mm_to_cm
                return 'cm', 'mm'
            elif lower_bound <= float(np.multiply(unknown, 4)) <= upper_bound:         #from_4th_to_cm
                return 'cm', '4thcm'
            elif lower_bound <= float(np.multiply(unknown, 2)) <= upper_bound:       #from_halfcm_to_cm
                return 'cm', 'halfcm'
            elif lower_bound <= float(np.multiply(unknown, 1)) <= upper_bound:      #from_cm_to_cm
                return 'cm', 'cm'
            else:
                return None, None

        elif self.is_standard:
            if lower_bound <= float(np.multiply(unknown, 32)) <= upper_bound:         #from_32nd_to_inch
                return 'inch', '32nd'
            elif lower_bound <= float(np.multiply(unknown, 16)) <= upper_bound:     #from_16th_to_inch
                return 'inch', '16th'
            elif lower_bound <= float(np.multiply(unknown, 8)) <= upper_bound:       #from_8th_to_inch
                return 'inch', '8th'
            elif lower_bound <= float(np.multiply(unknown, 4)) <= upper_bound:       #from_4th_to_inch
                return 'inch', '4th'
            elif lower_bound <= float(np.multiply(unknown, 2)) <= upper_bound:      #from_halfinch_to_inch
                return 'inch', 'halfinch'
            elif lower_bound <= float(np.multiply(unknown, 1)) <= upper_bound:      #from_inch_to_inch
                return 'inch', 'inch'
            else:
                return None, None