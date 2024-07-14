-- Function: SafeDiv
-- Description: Divides two integers and returns the result with one decimal place, or 0 if the divisor is 0
DELIMITER //

CREATE FUNCTION SafeDiv(a INT, b INT)
RETURNS DECIMAL(10, 1)
BEGIN
  DECLARE result DECIMAL(10, 1);

  IF b = 0 THEN
    SET result = 0.0;
  ELSE
    SET result = CAST(a AS DECIMAL(10, 1)) / b;
  END IF;

  RETURN result;
END //
DELIMITER ;

