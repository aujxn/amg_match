#[tokio::main]
pub async fn main() -> Result<(), Box<dyn std::error::Error>> {
    pretty_env_logger::init();
    amg_match::suitesparse_dl::download_all().await?;
    Ok(())
}
