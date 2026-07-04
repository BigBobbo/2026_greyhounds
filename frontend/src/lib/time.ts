/**
 * Backend timestamps are naive UTC strings (no offset). Parsing them with
 * plain `new Date(...)` treats them as LOCAL time, which skews staleness
 * math by the timezone offset (an hour in Irish summer time). Append Z
 * unless the string already carries timezone info.
 */
export function parseUtc(value: string | null | undefined): Date | null {
  if (!value) return null;
  const hasTz = /([zZ]|[+-]\d{2}:?\d{2})$/.test(value);
  return new Date(hasTz ? value : value + 'Z');
}
