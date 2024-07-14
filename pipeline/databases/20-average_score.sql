-- Procedure: ComputeAverageScoreForUser
-- Description: Computes and updates the average score for a given user
DELIMITER //

CREATE PROCEDURE ComputeAverageScoreForUser(
    IN user_id INT
)
BEGIN
    DECLARE avg_score FLOAT;

    -- Calculate average score
    SELECT AVG(score) INTO avg_score FROM corrections WHERE user_id = user_id;

    -- Update user's average_score in the users table
    UPDATE users SET average_score = avg_score WHERE id = user_id;
    
    -- Output for testing purposes
    SELECT "Average score computed and updated" AS msg;
END //

DELIMITER ;

