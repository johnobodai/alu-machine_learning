-- Function: SafeDiv
-- Description: Safely divides the first number by the second number or returns 0 if the second number is zero.
-- Arguments: a - INT, b - INT
-- Returns: DECIMAL(10, 6)
DELIMITER //

CREATE FUNCTION SafeDiv(a INT, b INT)
RETURNS DECIMAL(10, 6)
BEGIN
  DECLARE result DECIMAL(10, 1);

  IF b = 0 THEN
    SET result = 0;
  ELSE
    SET result = a / b;
  END IF;

  RETURN result;
END //

DELIMITER ;

