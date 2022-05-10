#[tokio::main]
pub async fn main() -> Result<(), Box<dyn std::error::Error>> {
    amg_match::suitesparse_dl::download_all().await?;
    Ok(())
}
